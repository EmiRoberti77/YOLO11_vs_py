"""
Emiliano Robert:
Object tracking trails:
File to detect object tracking using YOLO11
Available trackers Bot-SORT and ByteTrack
"""
from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

video_path_a="../Resources/Videos/video7.mp4"
video_path_b="../Resources/Videos/video8.mp4"

# Load YOLO model
model = YOLO("yolo11n.pt")
# Create a video Capture Object
cap = cv2.VideoCapture(video_path_a)
track_history = defaultdict(lambda : [])
while True:
  ret, frame = cap.read()
  if ret:
    # Run YOLO Tracking on the video frame
    # NOTE: persist=True tells the model to expect Tracks from the previous frame
    results = model.track(frame, persist=True)
    # Get the bounding box coordinates
    boxes = results[0].boxes.xywh.cpu()
    track_id = results[0].boxes.id.int().tolist()
    # Visualize the results on the frame\
    annotated_frame = results[0].plot()
    # Plot the tracks
    for box, track_id in zip(boxes, track_id):
      x, y, w, h = box
      track = track_history[track_id]
      track.append((float(x), float(y))) # x, y center point
      if len(track) > 30:
        track.pop(0)
      # Draw the tracking lines
      points = np.hstack(track).astype(np.int32).reshape(-1,1,2)
      cv2.polylines(annotated_frame,[points], isClosed=True, color=(230, 230, 230), thickness=10)
       
    # Display the annotated_frame
    cv2.imshow("Object Tracking", annotated_frame)
    # Break the loop if "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord("1"):
      break
  else:
    break

# Release all resources
cap.release()
cv2.destroyAllWindows()

