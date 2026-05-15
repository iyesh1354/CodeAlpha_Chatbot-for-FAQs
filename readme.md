# Object Detection and Tracking
### CodeAlpha Internship — Computer Vision Task

Real-time object detection using **YOLOv8** and tracking using the **SORT** algorithm (Kalman Filter + Hungarian Algorithm). Works with a webcam or any video file.

---

## Project Structure

```
CodeAlpha_ObjectDetection/
├── main.py              ← Main pipeline (detection + tracking + display)
├── sort.py              ← SORT tracker (Kalman Filter implementation)
├── requirements.txt     ← Python dependencies
└── README.md            ← This file
```

---

## How It Works

```
Webcam / Video
      │
      ▼
  Each Frame
      │
      ▼
 YOLOv8 Model  ──→  Bounding boxes + class labels + confidence
      │
      ▼
 SORT Tracker  ──→  Assigns unique tracking ID to each object
      │
      ▼
 Draw on Frame ──→  Colored box + "ID:3 person 94%"
      │
      ▼
 Display / Save
```

**YOLOv8 (You Only Look Once v8)** — detects 80 object classes in one forward pass.  
**SORT** — uses a Kalman Filter to predict object position and the Hungarian Algorithm to match detections across frames.

---

## Setup — Step by Step

### Step 1 — Install Python
Download Python 3.9+ from https://python.org  
Make sure to check **"Add Python to PATH"** during install.

### Step 2 — Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
> First run will auto-download `yolov8n.pt` (~6 MB) from Ultralytics.

---

## Running the Project

### Use webcam (default)
```bash
python main.py
```

### Use a video file
```bash
python main.py --source path/to/video.mp4
```

### Save output to output.mp4
```bash
python main.py --source path/to/video.mp4 --save
```

### Webcam + save
```bash
python main.py --source 0 --save
```

### Headless mode (no display window, only save)
```bash
python main.py --source video.mp4 --save --no-display
```

### Press **Q** or **ESC** in the window to quit.

---

## Output

Each detected object shows:
- Colored bounding box (unique color per tracking ID)
- Label: `ID:3 person 94%`
- HUD bar at top: source name, FPS, detection count, track count

---

## Adjusting Settings

Open `main.py` and edit these constants at the top:

| Constant | Default | Description |
|----------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | `0.4` | Lower = more detections, more noise |
| `NMS_THRESHOLD` | `0.45` | Non-max suppression overlap threshold |
| `MODEL_PATH` | `yolov8n.pt` | Change to `yolov8s.pt` for better accuracy |

### Available YOLO models (accuracy vs speed):
| Model | Speed | Accuracy |
|-------|-------|----------|
| `yolov8n.pt` | Fastest | Lower |
| `yolov8s.pt` | Fast | Good |
| `yolov8m.pt` | Medium | Better |
| `yolov8l.pt` | Slow | High |
| `yolov8x.pt` | Slowest | Best |

---

## SORT Tracker Parameters

In `main.py`, the tracker is initialised as:
```python
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_age` | `30` | Frames to keep a track alive without detection |
| `min_hits` | `3` | Frames needed before showing a track |
| `iou_threshold` | `0.3` | Overlap needed to match detection to track |

---

## Detectable Object Classes (80 total)

Person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird,
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard,
sports ball, kite, baseball bat, baseball glove, skateboard, surfboard,
tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl,
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza,
donut, cake, chair, couch, potted plant, bed, dining table, toilet,
TV, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear,
hair drier, toothbrush.

---

## Requirements

- Python 3.9+
- Webcam (for live mode) or a `.mp4` / `.avi` video file
- 4 GB RAM minimum (8 GB recommended)
- GPU optional but speeds things up significantly

---

## GitHub Upload

Name your repository exactly:
```
CodeAlpha_ObjectDetection
```

Upload these files:
```
main.py
sort.py
requirements.txt
README.md
```

Do NOT upload `.pt` model files (too large) — they auto-download.

---

## LinkedIn Post Template

> Built a real-time Object Detection & Tracking system using YOLOv8 + SORT algorithm as part of my @CodeAlpha internship!
>
> Features:
> - Real-time detection on webcam / video
> - Unique tracking IDs with Kalman Filter
> - 80 object classes detected
> - Smooth bounding boxes with labels and confidence scores
>
> GitHub: [your link here]
> #CodeAlpha #ComputerVision #Python #YOLOv8 #ObjectDetection #MachineLearning