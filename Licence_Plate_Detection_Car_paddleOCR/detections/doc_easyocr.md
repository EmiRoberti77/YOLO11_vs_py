# Licence Plate Detection and OCR Pipeline Using YOLO and EasyOCR

## Overview

This project combines a custom YOLOv11 object detection model and EasyOCR to accurately detect and read vehicle license plates from video files and a custom model trained to extract the area of the number plate. It includes video frame extraction, object detection, OCR processing, and video encoding of results. The code is implemented in Python using OpenCV, Ultralytics YOLO, and EasyOCR.

---

## Technologies Used

- **[YOLOv11](https://docs.ultralytics.com/):** Custom-trained model (`detection_plate_n_v1.pt`) for detecting license plates ( custom model )
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR):** Optical character recognition (OCR) engine used to extract alphanumeric text from plates.
- **[OpenCV](https://opencv.org/):** Used for video capture, image manipulation, drawing, and video encoding.

---

## File Structure

- `models/detection_plate_n_v1.pt` – Custom YOLO model for detecting plates.
- `Resources/Videos/car3.mp4` – Sample input video with visible license plates.
- `output_plates.avi` – Output video with detected plates annotated with OCR text.

---

## Components

### 1. `LicencePlateDetection`

Handles YOLO-based plate detection and EasyOCR-based text recognition from video frames.

#### Key Methods:

- **`__init__(model_path)`**: Loads YOLO model and EasyOCR reader.
- **`detect_frame(frame)`**:

  - Runs YOLO prediction.
  - Filters detections for "License_Plate" class.
  - Crops the plate, converts to grayscale.
  - Uses EasyOCR to extract text.
  - Annotates the frame with text and bounding box.

- **`detect_frames(frames)`**:

  - Applies detection to a list of frames.
  - Returns annotated frames that contain license plate detections.

### 2. `OAIXVidCap`

Handles video reading and writing, converting frames to/from video files.

#### Key Methods:

- **`extract_frames()`**:

  - Reads all frames from the input video.
  - Returns list of `frame` objects.

- **`encode_frames(frames)`**:

  - Encodes list of frames into an output `.avi` file.
  - Uses MJPG codec at 24fps.

- **`__del__()`**:

  - Releases video resources.

---

## How It Works

1. Video is loaded using OpenCV’s `cv2.VideoCapture`.
2. All video frames are extracted into a list.
3. A subset (or full list) of frames is passed to `LicencePlateDetection.detect_frames()`.
4. Each frame is analyzed:

   - YOLO detects bounding boxes for plates.
   - EasyOCR reads alphanumeric characters in those regions.
   - Annotations are drawn on the frame.

5. The final annotated frames are written to `output_plates.avi`.

---

## OCR Accuracy Considerations

- EasyOCR works well for Latin-alphabet plates.
- Converting the plate to grayscale before OCR improves accuracy.
- You may improve detection by tuning `conf` and image preprocessing (e.g., denoising, contrast).

---

## Usage

```bash
python3 easyocr.py
```

This loads `car3.mp4`, processes every frame for plate detection and OCR, then saves the result to `output_plates.avi`.

---

## Notes

- Ensure model path and input video are correct.
- Output file will be overwritten on each run.
- Adjust the number of processed frames by modifying the `starts` and `ends` indices.

---

## Future Improvements

- Add GUI or stream support.
- Batch OCR for multiple plates in a frame.
- Integrate multilingual OCR.
- Add confidence threshold filter for OCR reliability.

---

## Author

Emiliano Roberti

Custom YOLOv8 model trained specifically for license plate detection to maximize recognition accuracy in traffic scenes.
