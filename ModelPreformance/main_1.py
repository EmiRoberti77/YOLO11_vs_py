import os
import cv2
from ultralytics import YOLO

# Steps
# Read an Image using OpenCV
# Load YOLO11 and preform object detection
# Dataset classes reference https://gist.github.com/SrikarNamburu/0945de8f9a8714ec245dde3443e9d487
# Add the classes parameters ( in this case class=0 is person and class=5 bus)
# max_det = specify the max parameters ( 300 is max )
# Add NMS IOU 'iou'

#load the model
model = YOLO("yolo11n.pt")

#load image resources
image_path = "Resources/Images/image1.jpg"
if not os.path.exists(image_path):
  print(f"ERROR:{image_path} not found")
else:
  image = cv2.imread(image_path)
  result = model(image, save=True, conf=0.25, classes=[0,5], max_det=50)
  #cv2.imshow("Image", image)
  #print("press any key to close Image window")
  #cv2.waitKey(0)