import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
from rfdetr import RFDETRBase
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_rfdert_model(weights_path: Path) -> RFDETRBase:
    """
    Load an RFDETR model from the specified weights file.
    Raises FileNotFoundError if the weights file does not exist.
    """
    if not weights_path.is_file():
        logger.error(f"Model weights not found: {weights_path}")
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    logger.info(f"Loading RFDETR model from %s", weights_path)
    return RFDETRBase(pretrain_weights=str(weights_path))


def detect_and_annotate(
    image: "cv2.Mat",
    model: RFDETRBase,
    conf_threshold: float = 0.25,
) -> Tuple[List[Tuple[int, int, int, int, float]], "cv2.Mat"]:
    """
    Run inference on the image, filter by confidence threshold,
    and annotate the detections on the image.

    Returns a tuple of (results, annotated_image), where results is a list of
    (x1, y1, x2, y2, score).
    """
    if image is None:
        logger.error("Provided image is empty or invalid.")
        raise ValueError("Invalid image for detection.")

    logger.debug("Running inference with RFDETR...")
    raw_preds = model.inference(image)
    results: List[Tuple[int, int, int, int, float]] = []
    annotated = image.copy()

    for *box, score, _ in raw_preds:
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        results.append((x1, y1, x2, y2, float(score)))
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

    logger.info("Detected %d objects with score >= %.2f", len(results), conf_threshold)
    return results, annotated


def save_image(image: "cv2.Mat", output_path: Path) -> Path:
    """
    Save the image to disk, creating parent directories if needed.
    Returns the path where the image was saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), image)
    if not success:
        logger.error("Failed to write image to %s", output_path)
        raise IOError(f"Could not write image to {output_path}")

    logger.info("Annotated image saved to %s", output_path)
    return output_path


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and annotate CCCD cards using RFDETR."
    )
    parser.add_argument(
        "--image", "-i",
        type=Path,
        required=True,
        help="Path to input image file",
    )
    parser.add_argument(
        "--weights", "-w",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models" / "rf-dert" / "checkpoint_best_ema.pth",
        help="Path to RFDETR model weights",
    )
    parser.add_argument(
        "--conf-threshold", "-t",
        type=float,
        default=0.25,
        help="Confidence threshold for filtering detections",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Destination path for annotated output image",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()
    try:
        model = load_rfdert_model(args.weights)
    except Exception as e:
        logger.exception("Error loading model: %s", e)
        sys.exit(1)

    img = cv2.imread(str(args.image))
    try:
        results, annotated = detect_and_annotate(img, model, args.conf_threshold)
    except Exception as e:
        logger.exception("Detection failed: %s", e)
        sys.exit(1)

    output_path = args.output or args.image.parent / f"detected_{args.image.name}"
    try:
        save_image(annotated, output_path)
    except Exception as e:
        logger.exception("Saving annotated image failed: %s", e)
        sys.exit(1)

    logger.info("Process completed successfully with %d detections.", len(results))


if __name__ == "__main__":
    main()
