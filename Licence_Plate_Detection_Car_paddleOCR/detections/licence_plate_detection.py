from ultralytics import YOLO
import paddleocr
import cv2

class LicencePlateDetection:
  def __init__(self, model_path) -> None:
    self.model = YOLO(model=model_path)

  def detect_frame(self, frame):
    results = self.model.predict(frame)[0]
    id_name_dict = results.names
    licence_plate_list = []
    for box in results.boxes:
      r = box.xyxy.tolist()[0]
      cls_id = int(box.cls.tolist()[0])
      cls_name = id_name_dict[cls_id]
      if cls_name == "Licence_Plate":
        licence_plate_list.append(cls_name)
    return licence_plate_list
