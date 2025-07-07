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
model = YOLO("./weights/best.pt")

count = 0
# Class Name
classNameFT = ["Licence plate"]
# Save the video outputs
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)

output = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                         fps, 
                         (frame_width, frame_height))

# FPS
ptime = 0
ctime = 0

def add_fps_to_frame(frame, fps):
  cv2.putText(frame, f"fps:{str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

reader = easyocr.Reader(["en"], gpu=False)
def ocr_image(frame, x1, y1, x2, y2):
  plat_image = frame[y1:y2, x1:x2]
  gray = cv2.cvtColor(plat_image, cv2.COLOR_BGR2GRAY)
  result = reader.readtext(gray)
  text=""
  for res in result:
    if len(result) == 1:
      text = res[1]
    if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
      text = res[1]
  
  return str(text), plat_image 


while True:
  ret, frame = cap.read()
  if ret:
    count += 1
    # print(f"frame_count:{count}")
    results = model.predict(frame, conf=0.25, iou=0.2, verbose=False)
    for r in results:
      boxes = r.boxes
      for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2,y2), (255,0,0), 3)
        cls = int(box.cls[0])
        className=classNameFT[cls]
        conf = math.ceil(box.conf[0] * 100) / 100
        plate_label, plate_crop = ocr_image(frame, x1, y1, x2, y2)
        
        if plate_crop.shape[0] > 0 and plate_crop.shape[1] > 0:
          pip_img = cv2.resize(plate_crop, (150, 50))
          frame[10:60, frame.shape[1]-160:frame.shape[1]-10] = pip_img

        textSize = cv2.getTextSize(plate_label, 0, fontScale=0.5, thickness=2)[0]
        c2 = x1 + textSize[0], y1 - textSize[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, [255,0,0], -1)
        cv2.putText(frame, plate_label, (x1, y1 - 2), 0, 0.5, [255,255,255], thickness=1, lineType = cv2.LINE_AA)
    
    # calculate fps
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime 
    # add fps value to the frame
    label = add_fps_to_frame(frame, fps)
    cv2.putText(frame, f"fps:{str(int(fps))}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    # Write video frame
    output.write(frame)
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord("1"):
      break
  else:
    break

cap.release()
cv2.destroyAllWindows()



