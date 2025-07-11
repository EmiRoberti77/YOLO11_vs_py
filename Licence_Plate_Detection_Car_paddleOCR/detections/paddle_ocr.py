from paddleocr import PaddleOCR
from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "../models/detection_plate_n_v1.pt"
INPUT_FILE = "../../Resources/Videos/car3.mp4"
LICENCE_PLATE = "License_Plate"

class LicencePlateDetection:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    #self.ocr = PaddleOCR(use_angle_cls=True, lang='en')


  def detect_frames(self, frames):
    licence_plate_detection = []
    for f in frames:
      self.detect_frame(frame=f)

  
  def detect_frame(self, frame):
    results = self.model.predict(frame, conf=0.3)[0]
    licence_plates_detections = []
    id_names_dict = results.names
    cv2.imwrite(f"f_out_{time.time()}.jpg", frame)
    for box in results.boxes:
      r = box.xyxy.tolist()[0]
      cls_id = int(box.cls.tolist()[0])
      cls_name = id_names_dict[cls_id]

      if cls_name == LICENCE_PLATE:
        licence_plates_detections.append(r)
        x1, y1, x2, y2 = map(int, r)
        cropped_plate = frame[y1:y2, x1:x2]
        print((x1,y1), (x2,y2))


class OAIXVidCap():
  def __init__(self, input_file) -> None:
    self.input_file = input_file
    self.cap = cv2.VideoCapture(self.input_file)

  def extract_frames(self) -> list:
    frames_list = []
    while True: 
      ret, frame = self.cap.read()
      if not ret:
        print("failed to read frame")
        break
      else:
        frames_list.append(frame)
    return frames_list

    

if __name__ == "__main__":
  vid_cap = OAIXVidCap(INPUT_FILE)
  frames = vid_cap.extract_frames()
  ocr = LicencePlateDetection("../models/detection_plate_n_v1.pt")
  frames_length = len(frames)
  starts = int(frames_length/2)
  ends = int(starts + 2)
  print(f"frames_length:{frames_length}=starts:{starts} -> ends:{ends}")
  subset_frames = frames[starts:ends]
  print(subset_frames)
  ocr.detect_frames(frames=subset_frames)








