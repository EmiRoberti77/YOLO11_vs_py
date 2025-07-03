import os
import cv2
import math
from ultralytics import YOLO
from coco_class_names import cocoClassNames
import requests

# Steps
# Create video capture object with cv2
# Load YOLO11 and preform object detection
# Dataset classes reference https://gist.github.com/SrikarNamburu/0945de8f9a8714ec245dde3443e9d487
# Add the classes parameters ( in this case class=0 is person and class=5 bus)
# max_det = specify the max parameters ( 300 is max )
# Add NMS IOU 'iou' adjust iou if there is issue with duplicate bounding boxes per detection
# Add show True if you want image to appear
# Add save_txt = True to save detection in a text file
# Add save_crop = True to save just the detection cropped image

# Load the model
model = YOLO("yolo11s.pt")
#load video resource
video_path = "Resources/Videos/video5.mp4"
if not os.path.exists(video_path):
  print(f"ERROR:{video_path} not found")
else:
  
  cap = cv2.VideoCapture(video_path)
  
 
  count = 0
  while True:
    ret, image = cap.read()
    results = model(image, 
                 save=False,
                 conf=0.25, 
                 )
    if ret:
      count += 1
      print(count)
      # iterate through the results to extract 
      # bounding boxes
      for result in results:
        boxes = result.boxes
        # extra coordinates of every bounding box per object
        for box in boxes:
          x1, y1, x2, y2 = box.xyxy[0]
          print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
          # convert the tensor into integers
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          print(f"x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")
          # draw the bounding boxes
          cv2.rectangle(image, (x1,y1), (x2,y2),[255,0,0], 2)
          # create a label to display class and confidence
          classNameInt = int(box.cls[0])
          className = cocoClassNames[classNameInt]
          conf = math.ceil(box.conf[0] * 100) / 100
          label = className + ":" + str(conf)
          text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
          c2 = x1 + text_size[0], y1 - text_size[1] - 3
          cv2.rectangle(image, (x1, y1),c2,[255,0,0], -1)
          cv2.putText(image, label, (x1, y1 - 2), 0, 0.5, [255,255,255],thickness=1,lineType=cv2.LINE_AA)

      # Display image
      cv2.imshow("Image", image)
      if cv2.waitKey(1) & 0xFF == ord('1'):
        break

    else:
      print("Could not read",video_path)
      break


    