import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from utils.logger import setup_logger

logger = setup_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time face detection using MTCNN from FaceNet-pytorch"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="0",
        help="Video source (camera index or file path), default is 0 for webcam",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        help="Computation device, e.g., 'cpu' or 'cuda'. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Optional path to save annotated video."
    )
    parser.add_argument(
        "--display", action="store_true", help="Display video in a window"
    )
    parser.add_argument(
        "--confidence", "-t",
        type=float,
        default=0.90,
        help="Minimum confidence threshold for face detections",
    )
    return parser.parse_args()

class FaceDetector:
    """Detect faces in an image using MTCNN from FaceNet-pytorch."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Initializing MTCNN on device %s", self.device)
        self.detector = MTCNN(keep_all=True, device=self.device)

    def detect_faces(self, frame: cv2.Mat, threshold: float) -> List[Dict[str, float]]:
        """
        Detect faces in the given BGR frame.

        Returns a list of dicts with 'box' and 'confidence'.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        boxes, probs = self.detector.detect(img)
        results: List[Dict[str, float]] = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob < threshold:
                    continue
                x1, y1, x2, y2 = map(int, box)
                results.append({"box": [x1, y1, x2, y2], "confidence": float(prob)})
        logger.debug("Detected %d faces above threshold %.2f", len(results), threshold)
        return results


def annotate_frame(frame: cv2.Mat, detections: List[Dict[str, float]]) -> cv2.Mat:
    """
    Draw bounding boxes and confidence scores on the frame.
    """
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        conf = det["confidence"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{conf:.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return annotated


def main() -> None:
    args = parse_args()

    # Determine video source
    source = int(args.source) if args.source.isdigit() else str(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Failed to open video source: %s", args.source)
        sys.exit(1)

    detector = FaceDetector(device=args.device)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
        logger.info("Recording annotated video to %s", args.output)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream or failed frame capture.")
                break

            detections = detector.detect_faces(frame, args.confidence)
            annotated = annotate_frame(frame, detections)

            if args.display:
                cv2.imshow("Face Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User interrupted display.")
                    break

            if writer:
                writer.write(annotated)

    except Exception as e:
        logger.exception("Error during processing: %s", e)
    finally:
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        logger.info("Face detection process terminated.")

if __name__ == "__main__":
    main()
