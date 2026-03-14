"""
AI Detection Service — Real YOLO Model Integration

Your model: ai_service/pothole_best_backup.pt (YOLOv8 / Ultralytics)

The model is loaded ONCE at startup (lazy) and reused for every request.
No separate HTTP server needed — inference runs inside FastAPI itself.

To switch back to stub mode: set AI_MODEL_ENABLED=false in .env
To point to a different model file: set AI_MODEL_PATH in .env
"""
from typing import Dict, Any, List
import io
import os
import cv2
import numpy as np
from config import settings

# Model is loaded once and cached here
_model = None
_model_path: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pothole_best_backup.pt"
)
CONFIDENCE_THRESHOLD = 0.35   # Reduced noise as requested
DEDUP_DISTANCE_PX    = 35     # Distance for merging close detections
FRAME_SAMPLE_RATE    = 3


def _load_model():
    """Load YOLO model (lazy, called once on first inference)."""
    global _model
    if _model is not None:
        return _model

    # Try both names
    paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pothole_best.pt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pothole_best_backup.pt")
    ]
    
    for p in paths:
        if os.path.exists(p):
            try:
                from ultralytics import YOLO
                print(f"[AI] 🚀 Loading YOLO model from: {p}")
                _model = YOLO(p)
                print(f"[AI] ✅ Model loaded successfully from {p}")
                return _model
            except Exception as e:
                print(f"[AI] ❌ Failed to load model {p}: {e}")
    
    print("[AI] 🛑 CRITICAL: No valid model found! AI will be disabled.")
    return None


def _simple_severity(w: float, h: float, confidence: float) -> str:
    """Helper for video detections."""
    return classify_severity_from_detection([0, 0, w, h], confidence)


def classify_severity_from_detection(bbox: List[float], confidence: float) -> str:
    """
    Classify pothole severity based on bounding box area and confidence score.
    bbox = [x, y, w, h] where w and h are pixel dimensions.
    """
    area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
    if area > 5000 and confidence >= 0.85:
        return "high"
    elif area > 2000 or confidence >= 0.7:
        return "medium"
    return "low"


def _run_yolo_on_bytes(file_bytes: bytes) -> Dict[str, Any]:
    """Run real YOLO inference on image bytes with EXIF fix."""
    from PIL import Image, ImageOps
    model = _load_model()
    if model is None:
        print("[AI] ⚠️ Model missing: Providing error response (no stub).")
        return {"potholes": [], "error": "Model not loaded"}

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        print(f"[AI] Running inference on image size: {img.size}")
        results = model(img, verbose=True) # Turn on verbose for logs

        potholes = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()   # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                
                # Filter by threshold here manually just in case model params are ignored
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                # Convert xyxy → xywh
                x1, y1, x2, y2 = xyxy
                w = x2 - x1
                h = y2 - y1
                
                # SKY GATE: Potholes cannot be in the top 35% of a dashcam/phone image
                # This mathematically eliminates clouds/trees being marked as potholes.
                if y1 < (img.height * 0.35):
                    print(f"[AI] 🛡️ Sky Gate: Blocked impossible pothole at y={y1} (Top 35% of {img.height}px)")
                    continue

                bbox = [x1, y1, w, h]
                severity = classify_severity_from_detection(bbox, conf)
                potholes.append({
                    "bbox": [round(v, 1) for v in bbox],
                    "confidence": round(conf, 4),
                    "severity": severity,
                })

        print(f"[AI] Found {len(potholes)} potholes above {CONFIDENCE_THRESHOLD}")
        return {"potholes": potholes}

    except Exception as e:
        import traceback
        err_msg = f"[AI] Inference error: {str(e)}"
        print(err_msg)
        print(traceback.format_exc())
        return {"potholes": [], "error": err_msg}


# Stub detection removed to prevent fake sky potholes.


# ---------------------------------------------------------------------------
# Video Helpers (Deduplication, Drawing, Pipeline)
# ---------------------------------------------------------------------------
def _center(bbox: List[float]) -> tuple:
    return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

def _deduplicate(potholes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not potholes: return []
    unique = []
    for p in potholes:
        cx, cy = _center(p["bbox"])
        p_time = p.get("timestamp", 0.0)
        merged = False
        for i, u in enumerate(unique):
            ux, uy = _center(u["bbox"])
            u_time = u.get("timestamp", 0.0)
            if abs(p_time - u_time) > 1.0: continue
            import math
            dist = math.sqrt((cx-ux)**2 + (cy-uy)**2)
            if dist < DEDUP_DISTANCE_PX:
                if p["confidence"] > u["confidence"]: unique[i] = p
                merged = True
                break
        if not merged: unique.append(p)
    return unique

def _draw_detections(frame, detections):
    import cv2
    for det in detections:
        x, y, w, h = [int(v) for v in det["bbox"]]
        conf = det["confidence"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Pothole {conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def _process_video(file_bytes: bytes, ext: str) -> tuple:
    import cv2
    import numpy as np
    import tempfile
    import uuid
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_filename = f"{uuid.uuid4().hex}_annotated.mp4"
        out_path = os.path.join(settings.UPLOAD_DIR, out_filename)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened(): writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        all_detections, current_dets, frame_id, model = [], [], 0, _load_model()
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_id % FRAME_SAMPLE_RATE == 0 and model:
                # Frame orientation is handled by OpenCV Capture usually
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = model(rgb, verbose=False)
                current_dets = []
                for b in res[0].boxes:
                    conf = float(b.conf[0])
                    if conf < CONFIDENCE_THRESHOLD: continue
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    det = {
                        "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2-x1), 1), round(float(y2-y1), 1)],
                        "confidence": round(float(conf), 4),
                        "severity": _simple_severity(float(x2-x1), float(y2-y1), float(conf)),
                        "timestamp": round(float(frame_id / fps), 2)
                    }
                    current_dets.append(det)
                all_detections.extend(current_dets)
            writer.write(_draw_detections(frame.copy(), current_dets))
            frame_id += 1
        cap.release()
        writer.release()
        return (out_filename, _deduplicate(all_detections))
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)

async def run_detection(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    ext = (filename or "").lower().split(".")[-1]
    if ext in ("mp4", "mov", "avi", "mkv", "webm"):
        annotated, potholes = _process_video(file_bytes, ext)
        return {"potholes": potholes, "annotated_video": f"/uploads/{annotated}" if annotated else None}
    return _run_yolo_on_bytes(file_bytes)


def _extract_first_frame(video_bytes: bytes, ext: str) -> bytes | None:
    """Extract first frame from video using OpenCV if available."""
    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(video_bytes, np.uint8)
        cap = cv2.VideoCapture()
        # Write to temp file since cv2 doesn't support in-memory video
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        os.unlink(tmp_path)
        if not ret:
            return None
        _, buf = cv2.imencode(".jpg", frame)
        return buf.tobytes()
    except Exception as e:
        print(f"[AI] Frame extraction failed: {e}")
        return None

if __name__ == "__main__":
    # Test block
    import sys
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        if os.path.exists(test_path):
            with open(test_path, "rb") as f:
                content = f.read()
                import asyncio
                res = asyncio.run(run_detection(content, test_path))
                print(f"RESULTS: {res}")
        else:
            print(f"File not found: {test_path}")
