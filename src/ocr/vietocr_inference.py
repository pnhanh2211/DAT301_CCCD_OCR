import os
import sys
from pathlib import Path
import torch
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

def vietocr_inference(image_path: str):
    """Perform OCR inference on a single image using the default vgg_transformer model."""
    try:
        # Initialize VietOCR config with default vgg_transformer
        config = Cfg.load_config_from_name("vgg_transformer")
        config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
        config['predictor']['beamsearch'] = True

        # Initialize predictor
        detector = Predictor(config)

        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert from numpy.ndarray to PIL.Image
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Perform inference
        result = detector.predict(img_pil)
        logger.info(f"Extracted text: {result}")

        # Draw text on image (convert back to BGR for OpenCV)
        output_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        cv2.putText(output_img, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Save result image
        output_path = f"output_ocr_{Path(image_path).name}"
        cv2.imwrite(output_path, output_img)
        logger.info(f"OCR result saved to {output_path}")

        # Optional: Display image (uncomment to show)
        # cv2.imshow("OCR Result", output_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return result

    except Exception as e:
        logger.error(f"Error during OCR inference: {e}")
        raise

def main():
    # Example usage with a single image
    image_path = input("Enter the path to the image: ")
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Perform inference with default model
    vietocr_inference(image_path)

if __name__ == "__main__":
    main()