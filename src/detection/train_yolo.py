import os
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import wandb

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add parent directory to sys.path for utils import
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger
from utils.config import load_config

logger = setup_logger(__name__)

class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""
    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        """
        Args:
            patience (int): Number of epochs to wait for improvement.
            min_delta (float): Minimum change in loss to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False

    def __call__(self, validation_loss: float) -> bool:
        """Check if training should stop based on validation loss."""
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = validation_loss
            self.counter = 0
        return self.early_stop

class CCCDDirectoryDataset(Dataset):
    """Custom dataset for loading CCCD images and labels from a directory."""
    def __init__(self, data_path: str, data_cfg: Dict, split: str = "train", transform=None):
        """
        Args:
            data_path (str): Path to the data directory.
            data_cfg (dict): Data configuration with directory structures.
            split (str): Dataset split ("train", "val", or "test").
            transform: Optional image transformations.
        """
        self.data_path = data_path
        self.transform = transform
        self.split = split
        self.image_files = []
        self.label_files = []

        # Get directory paths from config
        self.img_dir = os.path.join(data_path, data_cfg[f"{split}_dir"])
        self.label_dir = os.path.join(data_path, data_cfg[f"{split}_annotations"])

        # Scan directory for images and labels
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")

        for img_file in os.listdir(self.img_dir):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(self.img_dir, img_file)
                label_file = os.path.join(self.label_dir, os.path.splitext(img_file)[0] + ".txt")
                if os.path.exists(label_file):
                    self.image_files.append(img_path)
                    self.label_files.append(label_file)
                else:
                    logger.warning(f"No label found for {img_file}")
        logger.info(f"Loaded {len(self.image_files)} {split} images from {data_path}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict:
        """Load and return a sample (image and labels) from directory."""
        try:
            # Load image
            img = Image.open(self.image_files[idx]).convert("RGB")
            img_shape = img.size

            # Load label
            labels = []
            with open(self.label_files[idx], "r") as f:
                label_content = f.read().strip().split("\n")
                for line in label_content:
                    if line.strip():
                        try:
                            label = list(map(float, line.split()))
                            if len(label) == 5:  # Ensure correct format: [class, x, y, w, h]
                                labels.append(label)
                            else:
                                logger.warning(f"Invalid label format in {self.label_files[idx]}: {line}")
                        except ValueError as e:
                            logger.error(f"Malformed label in {self.label_files[idx]}: {line}, error: {e}")
                            continue

            # Apply transforms
            if self.transform:
                img = self.transform(img)

            return {
                "img": img,
                "labels": torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5)),
                "img_size": img_shape,
                "path": self.image_files[idx],
            }
        except Exception as e:
            logger.error(f"Error loading sample {idx} from {self.image_files[idx]}: {e}")
            raise

def train_yolo(config_path: str = None, data_config_path: str = None) -> str:
    """
    Train a YOLOv11 model with a custom dataset and log to WandB.

    Args:
        config_path (str): Path to training configuration YAML file.
        data_config_path (str): Path to data configuration YAML file.

    Returns:
        str: Path to the saved best model weights.
    """
    # Set default config paths relative to project root
    base_dir = Path(__file__).parent.parent / ".." / "models" / "yolov11"
    config_path = config_path or str(base_dir / "train_config.yaml")
    data_config_path = data_config_path or str(base_dir / "cccd_data.yaml")

    # Load configurations
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        with open(data_config_path, "r") as f:
            data_cfg = yaml.safe_load(f)
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML format in config file: {e}")
        raise

    # Validate configurations
    if not config or not data_cfg:
        raise ValueError("Empty configuration file")

    # Extract WandB configuration
    wandb_config = config.get("wandb", {})
    wandb_project = wandb_config.get("project", "cccd-ocr-face-verification")
    wandb_entity = wandb_config.get("entity", None)
    wandb_group = wandb_config.get("group", None)
    wandb_job_type = wandb_config.get("job_type", "train")
    wandb_tags = wandb_config.get("tags", [])

    # Initialize WandB with configuration from YAML
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=wandb_tags,
        config=config
    )

    # Set up CUDA devices
    cuda_visible_devices = config.get("cuda_devices", "0")  # Sử dụng GPU 0 (RTX 3050)
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using device: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, falling back to CPU")

    # Initialize model
    pretrained_weights = config.get("pretrained_weights", "yolov11n.pt")
    try:
        model = YOLO(pretrained_weights)
        logger.info(f"Loaded pretrained weights from {pretrained_weights}")
    except Exception as e:
        logger.error(f"Failed to load pretrained weights {pretrained_weights}: {e}")
        raise

    # Create datasets
    try:
        train_dataset = CCCDDirectoryDataset(
            data_path=data_cfg["train"],
            data_cfg=data_cfg,
            split="train",
            transform=model.transforms,
        )
        val_dataset = CCCDDirectoryDataset(
            data_path=data_cfg["val"],
            data_cfg=data_cfg,
            split="val",
            transform=model.transforms,
        )
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        raise

    # Training configuration
    train_args = {
        "data": data_config_path,
        "epochs": config.get("epochs", 100),
        "imgsz": config.get("imgsz", 640),
        "batch": config.get("batch", 16),
        "device": [0] if torch.cuda.is_available() else ["cpu"],  # Sử dụng GPU 0
        "workers": config.get("workers", 8),
        "optimizer": config.get("optimizer", "AdamW"),
        "lr0": config.get("learning_rate", 0.01),
        "weight_decay": config.get("weight_decay", 0.0005),
        "momentum": config.get("momentum", 0.937),
        "warmup_epochs": config.get("warmup_epochs", 3),
        "close_mosaic": config.get("close_mosaic", 10),
        "project": config.get("project", "runs/train"),
        "name": config.get("name", "exp"),
        "save_period": config.get("save_period", 10),
        "cache": config.get("cache", False),
        "multi_scale": config.get("multi_scale", True),
        "amp": config.get("amp", True),
        "patience": config.get("early_stopping_patience", 7),
        "exist_ok": True,
    }

    # Train the model with WandB logging
    try:
        # Log training arguments to WandB
        wandb.config.update(train_args)

        results = model.train(**train_args)
        model_save_path = os.path.join(train_args["project"], train_args["name"], "weights/best.pt")
        logger.info(f"Training completed. Best model saved at: {model_save_path}")

        # Log final metrics to WandB
        final_metrics = results.results_dict
        logger.info(f"Final metrics: {final_metrics}")
        wandb.log(final_metrics)
        wandb.finish()

        return model_save_path

    except RuntimeError as e:
        logger.error(f"Training failed due to runtime error (e.g., OOM): {e}")
        wandb.finish()
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        wandb.finish()
        raise

if __name__ == "__main__":
    train_yolo()