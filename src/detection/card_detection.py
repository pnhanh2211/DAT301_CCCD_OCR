import sys
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
from rfdetr import RFDETRBase

# Thêm src vào sys.path để import utils.logger khi chạy trực tiếp
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from utils.logger import setup_logger

logger = setup_logger(__name__)

def load_rfdert_model(weights_path: Path) -> RFDETRBase:
    if not weights_path.is_file():
        logger.error(f"Model weights not found: {weights_path}")
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    logger.info(f"Loading RFDETR model from %s", weights_path)
    return RFDETRBase(pretrain_weights=str(weights_path))

def detect_and_annotate(
    image: "cv2.Mat",
    model: "RFDETRBase",
    conf_threshold: float = 0.25,
) -> Tuple[List[Tuple[int, int, int, int, float]], "cv2.Mat"]:
    if image is None:
        logger.error("Provided image is empty or invalid.")
        raise ValueError("Invalid image for detection.")
    logger.debug("Running inference with RFDETR...")
    detections = model.predict(image)
    results: List[Tuple[int, int, int, int, float]] = []
    annotated = image.copy()
    if hasattr(detections, 'xyxy') and hasattr(detections, 'confidence'):
        xyxy = detections.xyxy
        conf = detections.confidence
        for i in range(len(xyxy)):
            score = float(conf[i]) if conf is not None else None
            if score is None:
                logger.warning("RFDETR: Detected box with None score, skipping.")
                continue
            if score < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, xyxy[i])
            results.append((x1, y1, x2, y2, score))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{score:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    else:
        logger.warning("RFDETR: No valid detections or output format not supported.")
    logger.info("Detected %d objects with score >= %.2f", len(results), conf_threshold)
    return results, annotated

def save_image(image: "cv2.Mat", output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        logger.error("Failed to write image to %s", output_path)
        raise IOError(f"Could not write image to {output_path}")
    logger.info("Annotated image saved to %s", output_path)
    return output_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python card_detection.py <image_path> [--weights <weights_path>] [--conf <threshold>] [--output <output_path>]")
        sys.exit(1)
    img_path = Path(sys.argv[1])
    weights_path = Path("models/rf-dert/checkpoint_best_ema.pth")
    conf_threshold = 0.25
    output_path = img_path.parent / f"detected_{img_path.name}"
    # Parse optional args
    for i, arg in enumerate(sys.argv):
        if arg == "--weights" and i+1 < len(sys.argv):
            weights_path = Path(sys.argv[i+1])
        if arg == "--conf" and i+1 < len(sys.argv):
            conf_threshold = float(sys.argv[i+1])
        if arg == "--output" and i+1 < len(sys.argv):
            output_path = Path(sys.argv[i+1])
    try:
        model = load_rfdert_model(weights_path)
    except Exception as e:
        logger.exception("Error loading model: %s", e)
        sys.exit(1)
    img = cv2.imread(str(img_path))
    try:
        results, annotated = detect_and_annotate(img, model, conf_threshold)
    except Exception as e:
        logger.exception("Detection failed: %s", e)
        sys.exit(1)
    try:
        save_image(annotated, output_path)
    except Exception as e:
        logger.exception("Saving annotated image failed: %s", e)
        sys.exit(1)
    logger.info("Process completed successfully with %d detections.", len(results))

if __name__ == "__main__":
    main()
