
# 🚘 Vehicle Number Plate Detection

**Author: Emiliano Roberti**  
This project demonstrates real-time detection of vehicle number plates using a **custom-trained YOLO model** (`best.pt`) for improved accuracy, along with **EasyOCR** for optical character recognition of the plate text. The output includes a **picture-in-picture (PiP)** preview of the detected plate and a display of the recognized text directly on the video.

---

## 🎯 Objective

- Detect vehicle license plates in a video stream.
- Use OCR to extract plate numbers from bounding boxes.
- Overlay the text and plate image back onto the video in real time.
- Save the processed video with detection results.

---

## 📦 Requirements

Install dependencies:
```bash
pip install opencv-python ultralytics easyocr
```

Ensure you have:
- A trained YOLOv8-compatible model (`./weights/best.pt`) optimized to detect number plates.
- A sample video (`car4.mp4`) located in `../Resources/Videos/`.

---

## 🔧 Code Breakdown

### Imports
```python
import ultralytics
from ultralytics import YOLO
import easyocr
import cv2
import math, time
```
- `ultralytics`: For YOLOv8 object detection.
- `easyocr`: For Optical Character Recognition.
- `cv2`: OpenCV, used for video handling and frame processing.
- `math` and `time`: For rounding values and measuring FPS.

---

### Video and Model Initialization
```python
cap = cv2.VideoCapture("../Resources/Videos/car4.mp4")
model = YOLO("./weights/best.pt")
```
- Load the video to process.
- Load the custom YOLO model trained by Emiliano Roberti to detect license plates.

---

### Output Video Setup
```python
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
output = cv2.VideoWriter('output.avi', ...)
```
- Reads the dimensions and FPS of the source video.
- Configures the output video writer using MJPG codec.

---

### FPS Display Function
```python
def add_fps_to_frame(frame, fps):
    cv2.putText(...)
```
- Adds a textual FPS counter on the top-left of each frame.

---

### OCR Function
```python
reader = easyocr.Reader(["en"], gpu=False)
```
- Initializes OCR reader in CPU mode.

```python
def ocr_image(frame, x1, y1, x2, y2):
    ...
    return str(text), plat_image
```
- Crops and grayscale-converts the bounding box.
- Applies OCR to extract text.
- Returns the detected text and cropped image for PiP.

---

### Main Loop
```python
while True:
    ret, frame = cap.read()
```
- Reads the video frame-by-frame.

```python
results = model.predict(frame, conf=0.25, iou=0.2, verbose=False)
```
- Predicts objects in the frame using YOLO with confidence threshold `0.25` and IOU of `0.2`.

---

### Bounding Box Loop
```python
for box in boxes:
    ...
```
- Iterates over each detected bounding box.
- Extracts coordinates and draws rectangles.

---

### OCR and PiP Display
```python
plate_label, plate_crop = ocr_image(frame, x1, y1, x2, y2)
pip_img = cv2.resize(plate_crop, (150, 50))
frame[10:60, frame.shape[1]-160:frame.shape[1]-10] = pip_img
```
- Extracts license plate text and crops the image.
- Displays the cropped plate in the top-right corner (picture-in-picture).

---

### Overlaying Text
```python
cv2.putText(frame, plate_label, ...)
```
- Renders the recognized license plate number directly on the frame above the detected region.

---

### Finalizing the Frame
```python
output.write(frame)
cv2.imshow("frame", frame)
```
- Writes the processed frame to the video output.
- Displays the current frame with all overlays.

---

### Exit & Cleanup
```python
if cv2.waitKey(1) & 0xFF == ord("1"):
    break
...
cap.release()
cv2.destroyAllWindows()
```
- Press `"1"` to stop processing.
- Cleans up video resources.

---

## 📂 Output

- `output.avi`: Video with bounding boxes, OCR results, FPS counter, and PiP overlays.

---

## 🧠 Model Training Note

> This project uses a custom-trained YOLOv8 model created by **Emiliano Roberti**, specifically tuned for detecting vehicle number plates. The dataset included thousands of annotated license plate images from various regions and lighting conditions, allowing the model to perform reliably in real-world scenarios.

---

## 💡 Future Enhancements

- Export OCR results to a CSV or database.
- Support real-time camera feed (`cv2.VideoCapture(0)`).
- Region-specific OCR filters for multilingual plate formats.
- Add detection timestamps to each recognized plate.

---

## 📸 Example Frame

<img width="1105" alt="Screenshot 2025-07-07 at 19 17 43" src="https://github.com/user-attachments/assets/52ec8d4d-e0da-40d9-97f1-624405fc0d9d" />

<img width="1105" alt="Screenshot 2025-07-07 at 19 17 52" src="https://github.com/user-attachments/assets/4ed570fd-4b3e-46ee-bff6-b4ab89b285c4" />

---

## 🧑‍💻 License

This project is open for educational use by Emiliano Roberti. Contact for commercial licensing or collaboration.

