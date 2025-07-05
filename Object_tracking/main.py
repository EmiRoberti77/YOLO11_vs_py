"""
Emiliano Robert:
Object tracking:
File to detect object tracking using YOLO11
Available trackers Bot-SORT and ByteTrack
"""
from ultralytics import YOLO
import os
video_path="../Resources/Videos/video7.mp4"
if not os.path.exists(video_path):
  print("Err:file not found")
else:
  model = YOLO("yolo11n.pt")
  results = model.track(source=video_path, show=True, save=True)