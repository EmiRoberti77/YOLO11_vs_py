"""
Emiliano Robert:
Object tracking:
File to detect object tracking using YOLO11
Available trackers Bot-SORT and ByteTrack
"""
from ultralytics import YOLO
import os
video_path_a="../Resources/Videos/video7.mp4"
video_path_b="../Resources/Videos/video8.mp4"
if not os.path.exists(video_path_a) or not os.path.exists(video_path_b):
  print("Err:file not found")
else:
  model = YOLO("yolo11n.pt")
  # Tracking using Bot-SORT ( default tracker )
  results = model.track(source=video_path_a, 
                        show=True, 
                        save=True)
  # Tracking using Byte-Track ( default tracker )
  results = model.track(source=video_path_b,
                        tracker="bytetrack.yaml",
                        conf=0.2,
                        iou=0.3,
                        show=True, 
                        save=True)