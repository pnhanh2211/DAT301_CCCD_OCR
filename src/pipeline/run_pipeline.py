import os
from pathlib import Path
import sys

# Add project root and src to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.logger import setup_logger
from ocr.extract_fields import extract_fields_from_image

logger = setup_logger(__name__)

def run_pipeline(image_path: str, model_path: str, config_path: str) -> dict:
    """Run full OCR pipeline on a CCCD image.

    Parameters
    ----------
    image_path: str
        Path to the input image.
    model_path: str
        Path to the trained YOLO model.
    config_path: str
        Path to the YOLO configuration file.

    Returns
    -------
    dict
        Extracted fields with text and confidence.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info("Starting field extraction")
    results = extract_fields_from_image(image_path, model_path, config_path)
    logger.info("Pipeline completed")
    return results

def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    model_path = str(base_dir / "runs" / "train" / "exp" / "weights" / "best.pt")
    config_path = str(base_dir / "models" / "yolov11" / "cccd_data.yaml")

    image_path = input("Enter path to CCCD image: ")
    results = run_pipeline(image_path, model_path, config_path)
    print("Extracted fields:")
    for field, data in results.items():
        text = data.get("text", "")
        conf = data.get("confidence", 0.0)
        print(f"{field}: {text} (Confidence: {conf:.2f})")

if __name__ == "__main__":
    main()
