# Car Detection with YOLO and OpenCV

This project demonstrates how to use a custom-trained YOLOv11 model to detect cars in video frames, draw bounding boxes around them, and save the output video.

---

## 📁 Project Structure

```
car_detection_project/
├── main.py
├── detections.py
├── utils/
│   └── video_utils.py
├── yolo11n.pt
├── tracker_stubs/
│   └── car_detection.pkl
└── output_videos/
    └── output.avi
```

---

## 🔧 Dependencies

- Python 3.8+
- OpenCV (`cv2`)
- Ultralytics YOLOv11
- Pickle (built-in)

Install required packages:

```bash
pip install ultralytics opencv-python
```

---

## 📄 `detections.py`

This file defines the `CarDetection` class:

### `__init__`

```python
self.model = YOLO(model_path)
```

- Loads the YOLOv11 model from the provided path.

### `detect_frames()`

```python
car_detections = car_detector.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/car_detection.pkl")
```

- If `read_from_stub` is `True`, loads precomputed detection results from a pickle file.
- Otherwise, processes each frame with the YOLO model to detect cars.
- Saves detection results to the stub file if `stub_path` is provided.

### `detect_frame()`

```python
results = self.model.predict(frame, iou=0.1, conf=0.30)[0]
```

- Predicts bounding boxes using YOLO.
- Filters only the results classified as "car".

### `draw_bboxes()`

```python
cv2.rectangle(frame, (x1, y1), (x2, y2), ...)
```

- Draws a rectangle and label "Car" around each detected car.
- Returns the list of modified frames.

---

## 📄 `utils/video_utils.py`

### `read_video(video_path)`

- Opens the video using OpenCV.
- Extracts all frames into a list.

### `save_video(frames, output_path)`

- Saves the list of frames as a video file using OpenCV.
- Requires resolution and FPS from the original video.

---

## 🚀 `main.py`

### Description

This script runs the full car detection pipeline:

```python
from utils.video_utils import (read_video, save_video)
from detections import CarDetection

__MAIN__ = "__main__"

def main():
  input_video_path = "../Resources/Videos/car4.mp4"
  output_video_path = "output_videos/output.avi"

  # Step 1: Read video
  video_frames = read_video(input_video_path)

  # Step 2: Detect cars
  car_detector = CarDetection(model_path="yolo11n.pt")
  car_detections = car_detector.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/car_detection.pkl")

  # Step 3: Draw bounding boxes
  car_detector.draw_bboxes(video_frames, car_detections)

  # Step 4: Save output video
  save_video(video_frames, output_video_path)

if __name__ == __MAIN__:
  main()
```

---

## ✅ Output

- Processed video saved to: `output_videos/output.avi`
- Contains bounding boxes and labels drawn on detected cars.

---

## 📌 Notes

- Make sure the YOLO model file `yolo11n.pt` is in the root directory.
- Stub loading speeds up re-runs by avoiding redundant inference.
- Adjust `iou` and `conf` thresholds in `detect_frame()` if too few/many cars are detected.

---

## 📷 Example

If a car is detected:

```text
Draws bounding box:
[100.0, 150.0, 300.0, 400.0] => Draws rectangle and writes 'Car'
```

---

## 🔄 Next Steps

- Add support for other object types (e.g. bus, truck).
- Export detections as JSON or CSV.
- Add GUI or Streamlit interface.

---

## 🧠 Author

**Emiliano Roberti**

Custom YOLOv11 model trained for high-accuracy vehicle detection.

---

Happy Detecting! 🚗📹
