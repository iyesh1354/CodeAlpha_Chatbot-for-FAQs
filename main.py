"""
Object Detection and Tracking
CodeAlpha Internship — Computer Vision Task
Uses YOLOv8 + SORT for real-time detection and tracking
"""

import cv2
import numpy as np
import argparse
import time
import os
from ultralytics import YOLO
from sort import Sort


# ─── Configuration ────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.4      # Minimum detection confidence
NMS_THRESHOLD        = 0.45     # Non-maximum suppression threshold
MODEL_PATH           = "yolov8n.pt"   # YOLOv8 nano (auto-downloads)
FONT                 = cv2.FONT_HERSHEY_SIMPLEX

# Color palette for tracking IDs (BGR)
COLORS = [
    (255, 100,  50), ( 50, 200, 255), ( 50, 255, 100),
    (200,  50, 255), (255, 200,  50), ( 50, 100, 255),
    (100, 255, 200), (255,  50, 150), (150, 255,  50),
    ( 50, 255, 255), (255, 150,  50), (200, 200,  50),
]

def get_color(track_id: int):
    """Return a consistent color for a given tracking ID."""
    return COLORS[int(track_id) % len(COLORS)]


# ─── Drawing Utilities ─────────────────────────────────────────────────────────

def draw_box(frame, x1, y1, x2, y2, track_id, label, conf):
    """Draw a bounding box with label and tracking ID."""
    color = get_color(track_id)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    text = f"ID:{track_id} {label} {conf:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.55, 1)
    label_y = max(y1, th + 6)
    cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 6, label_y + 2), color, -1)

    # Label text (white)
    cv2.putText(frame, text, (x1 + 3, label_y - 3), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, fps, det_count, track_count, source_name):
    """Draw heads-up display with FPS and counts."""
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    info = (f"  {source_name}   |   FPS: {fps:.1f}   |   "
            f"Detections: {det_count}   |   Tracks: {track_count}   |   "
            f"Press Q to quit")
    cv2.putText(frame, info, (8, 25), FONT, 0.52, (220, 220, 220), 1, cv2.LINE_AA)


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def run(source, show_window=True, save_output=False):
    """
    Main detection + tracking loop.

    Parameters
    ----------
    source      : int (webcam index) or str (video file path)
    show_window : bool — display live window
    save_output : bool — save result to output.mp4
    """

    # Load YOLO model (downloads yolov8n.pt on first run ~6 MB)
    print("[INFO] Loading YOLOv8 model …")
    model = YOLO(MODEL_PATH)
    class_names = model.names          # {0: 'person', 1: 'bicycle', …}

    # Initialise SORT tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    source_name = "Webcam" if isinstance(source, int) else os.path.basename(str(source))

    # Optional output writer
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("output.mp4", fourcc, fps_src, (frame_w, frame_h))
        print("[INFO] Saving output to output.mp4")

    print(f"[INFO] Source : {source_name}  ({frame_w}×{frame_h} @ {fps_src:.0f} fps)")
    print("[INFO] Running — press Q in the window to stop.\n")

    fps_timer = time.time()
    fps_display = 0.0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video / no frame received.")
            break

        frame_count += 1

        # ── 1. YOLO Detection ────────────────────────────────────────────────
        results = model(frame, conf=CONFIDENCE_THRESHOLD,
                        iou=NMS_THRESHOLD, verbose=False)[0]

        # Build detection array [x1, y1, x2, y2, conf] for SORT
        detections = []
        det_labels = {}          # map box-index → class name

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = class_names.get(cls, str(cls))
            detections.append([x1, y1, x2, y2, conf])
            det_labels[i] = (label, conf)

        det_array = np.array(detections) if detections else np.empty((0, 5))

        # ── 2. SORT Tracking ─────────────────────────────────────────────────
        # Returns array of [x1, y1, x2, y2, track_id]
        tracks = tracker.update(det_array)

        # ── 3. Draw Results ──────────────────────────────────────────────────
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)

            # Match nearest detection label by IoU overlap
            best_label, best_conf = "object", 0.0
            best_iou = 0.0
            for idx, (lbl, cf) in det_labels.items():
                dx1, dy1, dx2, dy2 = map(int, detections[idx][:4])
                iou = _iou([x1,y1,x2,y2], [dx1,dy1,dx2,dy2])
                if iou > best_iou:
                    best_iou, best_label, best_conf = iou, lbl, cf

            draw_box(frame, x1, y1, x2, y2, track_id, best_label, best_conf)

        # ── 4. HUD ───────────────────────────────────────────────────────────
        elapsed = time.time() - fps_timer
        if elapsed >= 0.5:
            fps_display = frame_count / elapsed
            fps_timer   = time.time()
            frame_count = 0

        draw_hud(frame, fps_display, len(detections), len(tracks), source_name)

        # ── 5. Display / Save ────────────────────────────────────────────────
        if show_window:
            cv2.imshow("Object Detection & Tracking — CodeAlpha", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                print("[INFO] User quit.")
                break

        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ─── IoU Helper ───────────────────────────────────────────────────────────────

def _iou(a, b):
    """Intersection-over-Union for two boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 + SORT Object Detection & Tracking")
    parser.add_argument("--source", default="0",
                        help="0 = webcam  |  path/to/video.mp4")
    parser.add_argument("--save",   action="store_true",
                        help="Save output to output.mp4")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no window), useful on servers")
    args = parser.parse_args()

    # Convert "0","1",… to int for webcam
    source = int(args.source) if args.source.isdigit() else args.source

    run(source=source,
        show_window=not args.no_display,
        save_output=args.save)