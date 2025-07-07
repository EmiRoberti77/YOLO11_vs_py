"""
Detecting Vehicle Number plates 
author:Emiliano Roberti
"""
import ultralytics
from ultralytics import YOLO
import easyocr
import cv2
import math
import time

# Load the video file
cap = cv2.VideoCapture("../Resources/Videos/car3.mp4")

# Class Name
classNameFT = ["Licence plate"]

# Save the video outputs
count = 0

while True:
  ret, frame = cap.read()
  if ret:
    count += 1
    print(f"frame_count:{count}")
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("1"):
      break
  else:
    break

cap.release()
cv2.destroyAllWindows()



