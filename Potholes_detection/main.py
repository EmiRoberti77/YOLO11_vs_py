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
import cv2
from ultralytics import YOLO

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
    count += 1
    results = model(frame, save=True, conf=0.3)
    for r in results:
      boxes = r.boxes
      for box in boxes:
        x1, y1, x2, y2 =  map(int, box.xyxy[0])
        print(x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), [255,0,0], 2)
        frame_name = prefix + str(count) + ext
        cv2.imwrite(filename=frame_name, img=frame)
  else:
    print("Err: reading frames")
    break

cap.release()
cv2.destroyAllWindows()
