# Line-by-Line Code Explanation: Licence Plate Detection with OCR

This document provides a detailed explanation of each line and block in the OCR pipeline code that combines YOLO and EasyOCR for detecting and reading license plates from videos.

---

## Imports

```python
import easyocr
from ultralytics import YOLO
import cv2
import time
```

- `easyocr`: For optical character recognition.
- `YOLO` (Ultralytics): Used for object detection (license plates).
- `cv2`: OpenCV library for image and video handling.
- `time`: Used for timestamps (optional).

---

## Constants

```python
MODEL_PATH = "../models/detection_plate_n_v1.pt"
INPUT_FILE = "../../Resources/Videos/car3.mp4"
LICENCE_PLATE = "License_Plate"
```

- `MODEL_PATH`: File path to the YOLOv8 custom-trained model.
- `INPUT_FILE`: Path to the input video containing license plates.
- `LICENCE_PLATE`: The name/class used to identify license plate detections.

---

## Class: LicencePlateDetection

```python
class LicencePlateDetection:
```

Encapsulates the detection and OCR process for license plates.

### Constructor

```python
def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.ocr = easyocr.Reader(['en'])
```

- Loads the YOLO model.
- Initializes the EasyOCR reader with English language support.

### `detect_frames`

```python
def detect_frames(self, frames):
```

- Accepts a list of frames and runs detection/OCR on each.

```python
licence_plate_detection = []
for f in frames:
    detected, f = self.detect_frame(frame=f)
    if detected:
        licence_plate_detection.append(f)
```

- Stores frames where a plate is detected and annotated.
- Calls `detect_frame` per frame.

```python
return licence_plate_detection
```

- Returns the filtered, annotated list of frames.

### `detect_frame`

```python
def detect_frame(self, frame):
```

Processes one frame: detects plate, performs OCR, and returns result.

```python
results = self.model.predict(frame, conf=0.3, verbose=True)[0]
id_names_dict = results.names
```

- Performs inference with a confidence threshold of 0.3.
- Retrieves class name dictionary.

```python
detected = False
for box in results.boxes:
    detected = True
```

- Initializes detection flag.
- Iterates through detected bounding boxes.

```python
r = box.xyxy.tolist()[0]
cls_id = int(box.cls.tolist()[0])
cls_name = id_names_dict[cls_id]
```

- Extracts bounding box coordinates.
- Retrieves class ID and name.

```python
if cls_name == LICENCE_PLATE:
```

- Filters only license plate detections.

```python
x1, y1, x2, y2 = map(int, r)
cropped_plate = frame[y1:y2, x1:x2]
gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
```

- Crops the plate area.
- Converts it to grayscale to improve OCR.

```python
ocr_result = self.ocr.readtext(gray)
```

- Performs OCR on the cropped grayscale image.

```python
if len(ocr_result) > 0:
    bbox, text, prob = ocr_result[0]
    print(text)
```

- Extracts first OCR result if available.
- Prints detected text.

```python
cv2.putText(frame, f"{text}:{prob:.1f}", (x1,y1-10), ...)
```

- Overlays text and confidence score onto the frame.

```python
cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,0), 2)
```

- Draws bounding box around the plate.

```python
licence_plates_detections.append(frame)
return detected, frame
```

- Returns the annotated frame and detection flag.

---

## Class: OAIXVidCap

Handles video input/output.

### Constructor

```python
def __init__(self, input_file):
    self.input_file = input_file
    self.cap = cv2.VideoCapture(self.input_file)
```

- Loads video from file.

### `extract_frames`

```python
def extract_frames(self):
    frames_list = []
    while True:
        ret, frame = self.cap.read()
        if not ret:
            break
        else:
            frames_list.append(frame)
    return frames_list
```

- Reads and stores all frames from video.

### `encode_frames`

```python
def encode_frames(self, frames):
```

Encodes a list of frames into a video.

```python
width = frames[0].shape[1]
height = frames[0].shape[0]
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output_plates.avi', fourcc, 24, (width, height))
```

- Sets resolution, codec, and output file.

```python
for f in frames:
    out.write(f)
out.release()
cv2.destroyAllWindows()
```

- Writes each frame and closes the video file.

### Destructor

```python
def __del__(self):
    print("Descructor")
    self.cap.release()
```

- Ensures that video capture is released on object deletion.

---

## Main Block

```python
if __name__ == "__main__":
```

Executes the full pipeline:

```python
vid_cap = OAIXVidCap(INPUT_FILE)
frames = vid_cap.extract_frames()
ocr = LicencePlateDetection(MODEL_PATH)
```

- Loads video and model.

```python
frames_length = len(frames)
starts = int(frames_length/2)
ends = int(starts + 2)
subset_frames = frames[starts:ends]
```

- Selects a subset of frames for quick testing (can be modified).

```python
frames = ocr.detect_frames(frames)
vid_cap.encode_frames(frames)
del vid_cap
```

- Performs detection + OCR.
- Saves output video.
- Cleans up resources.

---

## Summary

This code demonstrates a full pipeline for video-based license plate OCR using YOLO and EasyOCR. It is modular, testable, and works well for datasets with clear plates. Future extensions may include real-time webcam support, improved OCR pre/post-processing, or vehicle tracking integration.
