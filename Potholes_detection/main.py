"""
Authour: Emiliano Roberti
This program is used to detect potholes in the road, can be used while a car is driving or a stand alone camera
Model - Yolo11
Dataset - custom trained 
"""
# steps
# import libs
# download model ( gdown "https://drive.google.com/uc?id=1iMitK9VCUWmBcZiiEPHK1d2pydALof6s&confirm=t" )
# run predictions

import ultralytics
import datetime
import cv2
from ultralytics import YOLO

_empty = '_'


pothole_events = []
class Base_Event:
  def __init__(self) -> None:
    self.timestamp = datetime.datetime.now().isoformat()
    self.frame_count = 0
    self.frame_name = _empty
class Pothole_event(Base_Event):
  def __init__(self) -> None:
    super().__init__()
    self.class_name = _empty
    self.bound_box = []
    


def appendResults(event:Pothole_event):
  pothole_events.append(event)
  return 0

video_path = "../Resources/videos/demo.mp4"
model = YOLO("potholes_v1.pt")
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "file not mounted in video capture"
prefix="frame_"
ext=".jpg"
count = 0
while True:
  ret, frame = cap.read()
  if ret:
    pothole_event_class = Pothole_event()
    count += 1
    results = model(frame, save=True, conf=0.3)
    for r in results:
      # print("names===>:",r.names)
      pothole_event_class.class_name = r.names
      # print("boxes===>:", r.boxes)
      boxes = r.boxes
      for box in boxes:
        # print("box===>:", box.xyxy)
        x1, y1, x2, y2 =  map(int, box.xyxy[0])
        pothole_event_class.bound_box = [x1, y1, x2, y2]
        pothole_event_class.frame_count = count
        # print(x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), [255,0,0], 2)
        frame_name = prefix + str(count) + ext
        cv2.imwrite(filename=frame_name, img=frame)
        appendResults(pothole_event_class)
    if count == 10:
      print("exit reached maxed count")
      break
  else:
    print("Err: reading frames")
    break

cap.release()
cv2.destroyAllWindows()

for event in pothole_events:
  # print(type(event))
  print("=====================")
  print(event.class_name)
  print(event.bound_box)
  print(event.timestamp)
  print(event.frame_count)
