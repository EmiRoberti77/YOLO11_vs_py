import easyocr
from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "../models/detection_plate_n_v1.pt"
INPUT_FILE = "../../Resources/Videos/car3.mp4"
LICENCE_PLATE = "License_Plate"

class LicencePlateDetection:
  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.ocr = easyocr.Reader(['en'])


  def detect_frames(self, frames):
    licence_plate_detection = []
    for f in frames:
      detected, f = self.detect_frame(frame=f)
      if detected:
        licence_plate_detection.append(f)
    
    return licence_plate_detection

  
  def detect_frame(self, frame):
    results = self.model.predict(frame, conf=0.3, verbose=True)[0]
    licence_plates_detections = []
    id_names_dict = results.names
    detected = False
    for box in results.boxes:
      detected = True
      r = box.xyxy.tolist()[0]
      cls_id = int(box.cls.tolist()[0])
      cls_name = id_names_dict[cls_id]

      if cls_name == LICENCE_PLATE:        
        x1, y1, x2, y2 = map(int, r)
        # crop the area where the plate is
        cropped_plate = frame[y1:y2, x1:x2]
        # convert plate to gray scale
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
        ocr_result = self.ocr.readtext(gray)
        if len(ocr_result) > 0:
          bbox, text, prob = ocr_result[0]
          print(text)
          cv2.putText(frame, f"{text}:{prob:.1f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Draw rectangle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,0), 2)
        licence_plates_detections.append(frame)

    return detected, frame


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
  
  def encode_frames(self, frames) -> None:
    if not len(frames):
      print("No frames to encode")
      return
    
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output_plates.avi', fourcc, 24, (width, height))
    try:
      print("starting encoding")
      for f in frames:
        out.write(f)
    except Exception as ex:
      print("Encoding error")
      print(ex)
    finally:
      out.release()
      cv2.destroyAllWindows()
  
  def __del__(self): 
    print("Descructor")
    self.cap.release()

    

if __name__ == "__main__":
  vid_cap = OAIXVidCap(INPUT_FILE)
  frames = vid_cap.extract_frames()
  ocr = LicencePlateDetection("../models/detection_plate_n_v1.pt")
  frames_length = len(frames)
  starts = int(frames_length/2)
  ends = int(starts + 2)
  print(f"frames_length:{frames_length}=starts:{starts} -> ends:{ends}")
  subset_frames = frames[starts:ends]
  # ocr.detect_frames(frames=subset_frames)
  frames = ocr.detect_frames(frames)
  vid_cap.encode_frames(frames)
  del vid_cap







