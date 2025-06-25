import os
import sys
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO
import cv2
import numpy as np

# Add parent directory to sys.path for utils import
sys.path.append(str(Path(__file__).parent.parent.parent))  # Thêm thư mục gốc (e:\cccd-ocr-face-verification)
sys.path.append(str(Path(__file__).parent.parent))        # Thêm thư mục src

from utils.logger import setup_logger


logger = setup_logger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def detect_cccd(image_path: str, model_path: str, config_path: str):
    """Perform detection on a single image using the YOLO model."""
    try:
        # Load configuration
        config = load_config(config_path)
        class_names = config["names"]

        # Load the trained model
        model = YOLO(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Perform inference
        results = model(image_path)

        # Process results
        for result in results:
            # Load image for drawing
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Draw bounding boxes and labels
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                label = f"{class_names[cls]}: {conf:.2f}"

                # Draw rectangle and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save or display the result
            output_path = f"output_{Path(image_path).name}"
            cv2.imwrite(output_path, img)
            logger.info(f"Detection result saved to {output_path}")

            # Optional: Display image (uncomment to show)
            # cv2.imshow("Detection Result", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise

def main():
    # Default paths (adjust as needed)
    base_dir = Path(__file__).parent.parent.parent  # Thư mục gốc: e:\cccd-ocr-face-verification
    model_path = str(base_dir / "runs/train/exp/weights/best.pt")
    config_path = str(base_dir / "models/yolov11/cccd_data.yaml")

    # Check if model and config exist
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Example usage with a single image
    image_path = input("Enter the path to the image: ")
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    detect_cccd(image_path, model_path, config_path)
if __name__ == "__main__":
    main()