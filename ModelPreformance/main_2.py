import os
import cv2
from ultralytics import YOLO

# Steps
# Read an Image using OpenCV
# Load YOLO11 and preform object detection
# Dataset classes reference https://gist.github.com/SrikarNamburu/0945de8f9a8714ec245dde3443e9d487
# Add the classes parameters ( in this case class=0 is person and class=5 bus)
# max_det = specify the max parameters ( 300 is max )
# Add NMS IOU 'iou' adjust iou if there is issue with duplicate bounding boxes per detection
# Add show True if you want image to appear
# Add save_txt = True to save detection in a text file
# Add save_crop = True to save just the detection cropped image

#load the model
model = YOLO("yolo11s.pt")

#load image resources
image_path = "Resources/Images/image3.png"
if not os.path.exists(image_path):
  print(f"ERROR:{image_path} not found")
else:
  image = cv2.imread(image_path)
  result = model(image, 
                 save=True, #save predictions
                 conf=0.25, #min confidence for prediction 
                 max_det=50, #max number of detection per object
                 iou=0.3, # the minimon intersection over union value to reduce multiple bounding boxes per object
                 show=True,  # display frame post processing ( good for debug )
                 save_txt=True, # save the output to a file
                 save_conf=True, # include the confidence score to output file
                 save_crop=True # in the prediction output save the cropped part od the images that have been detected
                 )