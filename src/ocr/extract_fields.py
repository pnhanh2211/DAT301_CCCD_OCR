import os
import sys
from pathlib import Path
import torch
import yaml
from ultralytics import YOLO
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
from PIL import Image
import numpy as np

# Add project root and utils directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Thư mục gốc
sys.path.append(str(Path(__file__).parent.parent))        # Thư mục src

from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def detect_fields(image_path: str, model_path: str, config_path: str) -> dict:
    """Detect fields on the CCCD image using YOLO."""
    try:
        # Load configuration
        config = load_config(config_path)
        class_names = config["names"]

        # Load the trained YOLO model
        model = YOLO(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Perform detection
        results = model(image_path)
        detected_fields = {}

        for result in results:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates
                conf = box.conf[0].item()  # Confidence score
                cls = int(box.cls[0].item())  # Class index
                field_name = class_names[cls]
                detected_fields[field_name] = {
                    "coords": (x1, y1, x2, y2),
                    "confidence": conf
                }
                logger.info(f"Detected {field_name} at {x1},{y1},{x2},{y2} with confidence {conf:.2f}")

            return detected_fields, img

    except Exception as e:
        logger.error(f"Error during field detection: {e}")
        raise

def extract_text_from_region(region_img: np.ndarray) -> str:
    """Extract text from a region using VietOCR."""
    try:
        # Initialize VietOCR config with default vgg_transformer
        config = Cfg.load_config_from_name("vgg_transformer")
        config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        config['predictor']['beamsearch'] = True
        detector = Predictor(config)

        # Convert numpy.ndarray to PIL.Image
        region_pil = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))

        # Perform inference
        result = detector.predict(region_pil)
        return result

    except Exception as e:
        logger.error(f"Error during text extraction: {e}")
        return ""

def extract_fields_from_image(image_path: str, model_path: str, config_path: str) -> dict:
    """Extract text from detected fields on a CCCD image."""
    try:
        # Detect fields
        detected_fields, img = detect_fields(image_path, model_path, config_path)

        # Extract text from each detected field
        extracted_data = {}
        for field_name, info in detected_fields.items():
            x1, y1, x2, y2 = info["coords"]
            # Crop the region
            region_img = img[y1:y2, x1:x2]
            if region_img.size == 0:
                logger.warning(f"Empty region for {field_name} at {x1},{y1},{x2},{y2}")
                extracted_data[field_name] = {"text": "", "confidence": info["confidence"]}
            else:
                text = extract_text_from_region(region_img)
                extracted_data[field_name] = {"text": text, "confidence": info["confidence"]}
                logger.info(f"Extracted text for {field_name}: {text}")

        # Draw results on image
        output_img = img.copy()
        for field_name, info in extracted_data.items():
            x1, y1, x2, y2 = detected_fields[field_name]["coords"]
            text = info["text"]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output_img, f"{field_name}: {text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save result image
        output_path = f"output_fields_{Path(image_path).name}"
        cv2.imwrite(output_path, output_img)
        logger.info(f"Field extraction result saved to {output_path}")

        return extracted_data

    except Exception as e:
        logger.error(f"Error during field extraction: {e}")
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
    image_path = input("Enter the path to the CCCD image: ")
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Extract fields
    results = extract_fields_from_image(image_path, model_path, config_path)
    print("Extracted fields:")
    for field, data in results.items():
        print(f"{field}: {data['text']} (Confidence: {data['confidence']:.2f})")

if __name__ == "__main__":
    main()