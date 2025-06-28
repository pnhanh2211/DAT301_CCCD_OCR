import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO
import cv2

# Add project root and src directories to PYTHONPATH
ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = Path(__file__).parent.parent
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def detect_cccd(image_path: str, model_path: str, config_path: str) -> None:
    """Perform detection on a single image using the YOLO model."""
    try:
        # Load config and model
        config = load_config(config_path)
        class_names = config.get("names", {})
        model = YOLO(model_path)
        logger.info("Loaded model from %s", model_path)

        # Run inference
        results = model(image_path)

        for result in results:
            # Load image for drawing
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            # Draw boxes and labels
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{class_names.get(cls, cls)}: {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Save annotated result
            output_path = f"output_{Path(image_path).name}"
            cv2.imwrite(output_path, img)
            logger.info("Detection result saved to %s", output_path)

    except Exception as e:
        logger.error("Error during detection: %s", e)
        raise


def main() -> None:
    base_dir = Path(__file__).parent.parent.parent
    model_path = str(base_dir / "runs" / "train" / "exp" / "weights" / "best.pt")
    config_path = str(base_dir / "models" / "yolov11" / "cccd_data.yaml")

    if not os.path.exists(model_path):
        logger.error("Model file not found: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        raise FileNotFoundError(f"Config file not found: {config_path}")

    image_path = input("Enter the path to the image: ")
    if not os.path.exists(image_path):
        logger.error("Image file not found: %s", image_path)
        raise FileNotFoundError(f"Image file not found: {image_path}")

    detect_cccd(image_path, model_path, config_path)

if __name__ == "__main__":
    main()