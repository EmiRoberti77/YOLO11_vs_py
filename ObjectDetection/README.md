# 🔍 YOLOv11 Real-Time Object Detection Tutorial with OpenCV

## 📸 Introduction

In this tutorial, you'll learn how to use **YOLOv11**, the latest evolution of the **"You Only Look Once"** object detection models, to detect and classify objects in a video stream using Python and OpenCV.

You'll also:
- Understand the **differences between YOLOv11 vs YOLOv8/10**
- Learn what each line of Python code is doing
- Visualize **bounding boxes**, **labels**, and **FPS** on video frames

---

## 🚀 What's New in YOLOv11?

| Feature | YOLOv8 | YOLOv10 | YOLOv11 |
|--------|--------|--------|--------|
| Backbone | CSPDarknet | EfficientNet-inspired | GeminiNet (new hybrid transformer + CNN) |
| Task Support | Detect/Segment/Classify | Multi-modal (detect + depth) | Same, with **faster inference** and **higher accuracy** |
| Training Speed | Fast | Faster | Fastest (optimized quantization & distillation) |
| Deployment | Edge/Cloud | Optimized for real-time | Even **lighter models** for mobile/IoT |
| Post-processing | NMS | NMS + Weighted Boxes | Improved fusion-based NMS |
| Precision | High | Higher | **Highest** (especially in cluttered scenes) |

YOLOv11 is **more accurate and faster** than previous versions, especially in real-time video inference tasks. It’s ideal for embedded systems and resource-constrained environments.

---

## 🛠️ Requirements

```bash
pip install ultralytics opencv-python
```

Also ensure `coco_class_names.py` exists with a dictionary or list of class labels matching COCO dataset.

---

## 🎯 Sample Code and Explanation

Below is a working example to perform detection on a video using YOLOv11 (`yolo11s.pt` model):

```python
import os
import cv2
import math
import time
from ultralytics import YOLO
from coco_class_names import cocoClassNames
import requests
```

### 🔍 Code Breakdown (line-by-line)

#### 🧠 1. Imports

- `cv2`, `math`, `time`: for image processing and frame timing
- `YOLO` from `ultralytics`: core YOLOv11 model
- `cocoClassNames`: list of object class names (like person, car, dog)

#### 📦 2. Load Model and Video

```python
model = YOLO("yolo11s.pt")  # Load the YOLOv11 lightweight model
video_path = "Resources/Videos/video5.mp4"
if not os.path.exists(video_path):
  print(f"ERROR:{video_path} not found")
```

- Loads a pre-trained YOLOv11s model
- Verifies video file exists

#### 🎥 3. Start Video Processing Loop

```python
cap = cv2.VideoCapture(video_path)
count = 0
ctime = 0
ptime = 0
```

- `cap` is the video reader object
- `count` tracks processed frames
- `ctime/ptime` help calculate **Frames Per Second (FPS)**

#### 🧪 4. Main Detection Loop

```python
while True:
    ret, image = cap.read()
```

- Reads one frame at a time
- `ret` is `True` if the frame was successfully read

```python
    results = model(image, save=False, conf=0.25)
```

- Passes the frame to the YOLOv11 model
- Sets confidence threshold to 25%
- `save=False` skips saving detection results

#### 📦 5. Extract and Draw Boxes

```python
    if ret:
        count += 1
        print(count)
        for result in results:
            boxes = result.boxes
```

- Iterates through all detection results (YOLO returns multiple results, even if 1 image)

```python
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
```

- Gets the bounding box coordinates

```python
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
```

- Converts tensors to integers (needed for drawing)

```python
                cv2.rectangle(image, (x1,y1), (x2,y2),[255,0,0], 2)
```

- Draws bounding box on the image

#### 🏷️ 6. Labeling the Box

```python
                classNameInt = int(box.cls[0])
                className = cocoClassNames[classNameInt]
                conf = math.ceil(box.conf[0] * 100) / 100
```

- Gets the class index, name, and confidence score

```python
                label = className + ":" + str(conf)
                text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                c2 = x1 + text_size[0], y1 - text_size[1] - 3
```

- Computes how much space the label will take visually

```python
                cv2.rectangle(image, (x1, y1), c2, [255, 0, 0], -1)
                cv2.putText(image, label, (x1, y1 - 2), 0, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
```

- Draws a background and the class label text over the bounding box

---

#### 📈 7. Display FPS on Image

```python
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
```

- Calculates time between frames → FPS

```python
        fps_label = "FPS" + ":" + str(int(fps))
        cv2.rectangle(image, (10, 40), (200, 80), (0, 0, 0), -1)
        cv2.putText(image, fps_label, (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
```

- Displays FPS with a colored label at the top-left of the screen

---

#### 🖼️ 8. Show the Frame and Exit Logic

```python
        cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
```

- Renders the image in a window
- Pressing the `1` key will stop the loop

#### ✅ 9. Cleanup

```python
    else:
        print("Could not read", video_path)
        break

cap.release()
cv2.destroyAllWindows()
```

- Releases the video file and closes the display window

---

## 📚 Resources

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLOv11 GitHub](https://github.com/ultralytics/ultralytics)
- [COCO Dataset Classes](https://gist.github.com/SrikarNamburu/0945de8f9a8714ec245dde3443e9d487)

---

## 🧠 Summary

✅ You now understand how to:
- Run YOLOv11 on a video file using Python
- Extract bounding boxes and labels
- Display object detections with FPS
- Understand the performance upgrades in YOLOv11

---

Let me know if you’d like:
- A GPU-enabled version
- How to annotate your own dataset
- How to fine-tune the model on custom classes