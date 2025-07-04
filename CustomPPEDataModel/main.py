import ultralytics
import cv2
import time
from ultralytics import YOLO


video_path = 'PPE_Part2.mp4'
model = YOLO('best.pt')
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Failed to open video model"

_ = model.predict(cv2.imread("image.jpg"), conf=0.3)

while True:
  ret, image = cap.read()
  start_time = time.time()
  results = model(image, save=False, verbose=False, conf=0.3)
  if ret: 
    for result in results:
      boxes = result.boxes
      for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2,y2), [255,0,0], 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    label = f"FPS:{fps:.2f}"
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Image", image) 
    if cv2.waitKey(1) & 0xFF == ord("1"):
      break
  else:
    print("Err")
    break

cap.release()
cv2.destroyAllWindows()
