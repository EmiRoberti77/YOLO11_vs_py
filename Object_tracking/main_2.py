"""
Emiliano Robert:
Object tracking:
File to detect object tracking using YOLO11
Available trackers Bot-SORT and ByteTrack
"""
from ultralytics import YOLO
import os
import cv2
video_path_a="../Resources/Videos/video7.mp4"
video_path_b="../Resources/Videos/video8.mp4"
if not os.path.exists(video_path_a) or not os.path.exists(video_path_b):
  print("Err:file not found")
else:
  # Load YOLO model
  model = YOLO("yolo11n.pt")
  # Create a video Capture Object
  cap = cv2.VideoCapture(video_path_a)
  while True:
    ret, frame = cap.read()
    if ret:
      # Run YOLO Tracking on the video frame
      # NOTE: persist=True tells the model to expect Tracks from the previous frame
      results = model.track(frame, persist=True)
      # Visualize the results on the frame
      annotated_frames = results[0].plot()
      # Display the annotated_frame
      cv2.imshow("Object Tracking", annotated_frames)
      # Break the loop if "q" is pressed
      if cv2.waitKey(1) & 0xFF == ord("1"):
        break
    else:
      break
  
  # Release all resources
  cap.release()
  cv2.destroyAllWindows()

