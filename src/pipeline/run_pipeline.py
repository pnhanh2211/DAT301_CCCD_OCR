import os
import sys
import time
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root and src to PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = Path(__file__).resolve().parents[1]
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from utils.logger import setup_logger
from detection.card_detection import load_rfdert_model, detect_and_annotate
from detection.face_detection import FaceDetector, annotate_frame
from ultralytics import YOLO
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
from facenet_pytorch import InceptionResnetV1

logger = setup_logger(__name__)

def clamp(val, low, high):
    return max(low, min(val, high))


def init_models(card_weights, yolo_weights, ocr_config, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # RF-DETR for card detection
    if not os.path.isfile(card_weights):
        raise FileNotFoundError(f'Card model checkpoint not found: {card_weights}')
    card_model = load_rfdert_model(Path(card_weights))

    # YOLO for field detection
    yolo_model = YOLO(str(yolo_weights))

    # MTCNN for face detection
    face_detector = FaceDetector(device=device)

    # Inception for face embedding
    face_recognizer = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # VietOCR for text recognition
    cfg = Cfg.load_config_from_name(ocr_config)
    cfg['device'] = device
    ocr_model = Predictor(cfg)

    return card_model, yolo_model, face_detector, face_recognizer, ocr_model, device


def compute_face_embedding(img, face_detector, face_recognizer):
    """Detect first face in img, compute normalized embedding"""
    if img is None:
        return None
    faces = face_detector.detect_faces(img, threshold=0)
    if not faces:
        return None
    x1, y1, x2, y2 = faces[0]['box']
    h, w = img.shape[:2]
    x1, y1 = clamp(int(x1), 0, w), clamp(int(y1), 0, h)
    x2, y2 = clamp(int(x2), 0, w), clamp(int(y2), 0, h)
    if x2 <= x1 or y2 <= y1:
        return None
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None
    face = cv2.resize(face_crop, (160, 160))
    device = next(face_recognizer.parameters()).device
    tensor = torch.from_numpy(face).permute(2,0,1).float().to(device) / 255.0
    emb = face_recognizer(tensor.unsqueeze(0))
    emb = emb.detach().cpu().numpy().flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else None


def compare_faces(emb1, emb2):
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2))


def extract_card_fields(card_img, yolo_model, ocr_model):
    """Detect field regions on card and OCR them"""
    results = yolo_model(card_img)
    fields = {}
    for res in results:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = clamp(x1, 0, card_img.shape[1]), clamp(y1, 0, card_img.shape[0])
            x2, y2 = clamp(x2, 0, card_img.shape[1]), clamp(y2, 0, card_img.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            region = card_img[y1:y2, x1:x2]
            pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            text = ocr_model.predict(pil)
            cls = int(box.cls[0].item())
            name = yolo_model.names[cls]
            fields[name] = text
            logger.debug(f'Field {name}: {text}')
    return fields


def run_image_pipeline(args, models):
    card_model, yolo_model, face_detector, face_recognizer, ocr_model, device = models
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {args.image}')
    H, W = img.shape[:2]
    records = []

    card_boxes, annotated = detect_and_annotate(img, card_model, args.conf_threshold)
    live_faces = face_detector.detect_faces(img, args.face_threshold)

    for idx, (x1,y1,x2,y2,conf) in enumerate(card_boxes):
        x1, y1 = clamp(int(x1),0,W), clamp(int(y1),0,H)
        x2, y2 = clamp(int(x2),0,W), clamp(int(y2),0,H)
        if x2<=x1 or y2<=y1:
            continue
        card_img = img[y1:y2, x1:x2]
        record = {
            'image_id': Path(args.image).stem,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'card_box': [x1,y1,x2,y2],
            'confidence': float(conf),
            'fields': {},
            'similarity': None,
            'allowed': None,
            'paths': {}
        }
        # save crops
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        card_path = out_dir / f'card_{idx}.jpg'
        cv2.imwrite(str(card_path), card_img)
        record['paths']['card_crop'] = str(card_path)

        # OCR fields
        record['fields'] = extract_card_fields(card_img, yolo_model, ocr_model)

        # embeddings
        card_emb = compute_face_embedding(card_img, face_detector, face_recognizer)
        live_emb = None
        if live_faces:
            fx1,fy1,fx2,fy2 = live_faces[0]['box']
            fx1,fy1 = clamp(int(fx1),0,W), clamp(int(fy1),0,H)
            fx2,fy2 = clamp(int(fx2),0,W), clamp(int(fy2),0,H)
            live_crop = img[fy1:fy2, fx1:fx2]
            live_path = out_dir / f'live_{idx}.jpg'
            cv2.imwrite(str(live_path), live_crop)
            record['paths']['live_face_crop'] = str(live_path)
            live_emb = compute_face_embedding(live_crop, face_detector, face_recognizer)
        sim = compare_faces(card_emb, live_emb)
        record['similarity'] = sim
        if sim < args.low_threshold:
            record['allowed'] = 'unknown'
        else:
            record['allowed'] = bool(sim >= args.sim_threshold)

        # annotate
        color = (0,255,0) if record['allowed'] else (0,0,255)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated, f"sim:{sim:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        records.append(record)

    # save JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f'Saved JSON results to {args.output}')

    cv2.imshow('Image Pipeline Result', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_realtime_pipeline(args, models):
    card_model, yolo_model, face_detector, face_recognizer, ocr_model, device = models
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera index {args.camera_index}')
    records = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning('Frame read failed, retry...')
            continue
        frame_count += 1
        H, W = frame.shape[:2]

        card_boxes, annotated = detect_and_annotate(frame, card_model, args.conf_threshold)
        live_faces = face_detector.detect_faces(frame, args.face_threshold)

        for idx, (x1,y1,x2,y2,conf) in enumerate(card_boxes):
            x1, y1 = clamp(int(x1),0,W), clamp(int(y1),0,H)
            x2, y2 = clamp(int(x2),0,W), clamp(int(y2),0,H)
            if x2<=x1 or y2<=y1:
                continue
            card_img = frame[y1:y2, x1:x2]
            record = {
                'image_id': f'realtime_{frame_count}',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'frame': frame_count,
                'card_box': [x1,y1,x2,y2],
                'confidence': float(conf),
                'fields': {},
                'similarity': None,
                'allowed': None,
                'paths': {}
            }
            # save crops
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            card_path = out_dir / f'realtime_card_{frame_count}_{idx}.jpg'
            cv2.imwrite(str(card_path), card_img)
            record['paths']['card_crop'] = str(card_path)

            record['fields'] = extract_card_fields(card_img, yolo_model, ocr_model)

            card_emb = compute_face_embedding(card_img, face_detector, face_recognizer)
            live_emb = None
            if live_faces:
                fx1,fy1,fx2,fy2 = live_faces[0]['box']
                fx1,fy1 = clamp(int(fx1),0,W), clamp(int(fy1),0,H)
                fx2,fy2 = clamp(int(fx2),0,W), clamp(int(fy2),0,H)
                live_crop = frame[fy1:fy2, fx1:fx2]
                live_path = out_dir / f'realtime_live_{frame_count}_{idx}.jpg'
                cv2.imwrite(str(live_path), live_crop)
                record['paths']['live_face_crop'] = str(live_path)
                live_emb = compute_face_embedding(live_crop, face_detector, face_recognizer)
            sim = compare_faces(card_emb, live_emb)
            record['similarity'] = sim
            if sim < args.low_threshold:
                record['allowed'] = 'unknown'
            else:
                record['allowed'] = bool(sim >= args.sim_threshold)

            color = (0,255,0) if record['allowed'] else (0,0,255)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            cv2.putText(annotated, f"sim:{sim:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            records.append(record)

        cv2.imshow('Realtime Pipeline', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # save JSON
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f'Saved JSON results to {args.output}')


def parse_args():
    p = argparse.ArgumentParser(description='CCCD OCR & Face Verification Pipeline')
    p.add_argument('--weights', '-w', default=str(ROOT_DIR/'runs'/'train'/'exp'/'weights'/'best.pt'), help='YOLO weights path')
    p.add_argument('--card-model', '-cm', default=str(ROOT_DIR/'models'/'rf-dert'/'checkpoint_best_ema.pth'), help='RF-DETR card model path')
    p.add_argument('--ocr-config', '-oc', default='vgg_transformer', help='VietOCR config')
    p.add_argument('--device', '-d', default=None, help='Device (cpu or cuda)')
    p.add_argument('--conf-threshold', '-t', type=float, default=0.25, help='Card detection confidence')
    p.add_argument('--face-threshold', '-ft', type=float, default=0.9, help='Face detection confidence')
    p.add_argument('--low-threshold', type=float, default=0.3, help='Threshold below which similarity is unknown')
    p.add_argument('--sim-threshold', '-st', type=float, default=0.6, help='Similarity threshold to allow')
    p.add_argument('--image', '-img', help='Path to input image (mutually exclusive with realtime)')
    p.add_argument('--camera-index', '-ci', type=int, default=0, help='Camera index for realtime')
    p.add_argument('--output', '-o', default=str(Path.cwd()/'results.json'), help='Output JSON file path')
    p.add_argument('--output-dir', default=str(Path.cwd()/'results'), help='Directory to save crops')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    models = init_models(args.card_model, args.weights, args.ocr_config, args.device)
    if args.image:
        run_image_pipeline(args, models)
    else:
        run_realtime_pipeline(args, models)
