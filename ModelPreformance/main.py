import os
import cv2
from ultralytics import YOLO

# Steps
# Read an Image using OpenCV
# Load YOLO11 and preform object detection

#load the model
model = YOLO("yolo11n.pt")

#load image resources
image_path = "Resources/Images/image1.jpg"
if not os.path.exists(image_path):
  print(f"ERROR:{image_path} not found")
else:
  image = cv2.imread(image_path)
  result = model(image, save=True)
  #cv2.imshow("Image", image)
  #print("press any key to close Image window")
  #cv2.waitKey(0)