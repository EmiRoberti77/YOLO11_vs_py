"""
Emiliano Robert:
Object tracking trails:
File to detect multithread object tracking using YOLO11
track objects in multiple video streams
Available trackers Bot-SORT and ByteTrack
"""
from ultralytics import YOLO
import threading
import cv2
from collections import defaultdict
import numpy as np

video_path_a="../Resources/Videos/video7.mp4"
video_path_b="../Resources/Videos/video8.mp4"
MODEL_NAMES = ["yolo11n.pt","yolo11n-seg.pt"]
SOURCES = [video_path_a, video_path_b]

def run_tracker_in_thread(model_name, file_name):
  #Run YOLO Tracker in its own thread for concurrent processing
  model = YOLO(model_name)
  results = model.track(source = file_name, save = True, stream = True, show=True)
  for r in results:
    pass
  

tracker_threads = []

for model_name, file_name in zip(MODEL_NAMES, SOURCES):
  thread = threading.Thread(target=run_tracker_in_thread, args=(model_name, file_name), daemon=True)
  tracker_threads.append(thread)
  thread.start()


# Wait for the all the treads to finish
for t in tracker_threads:
  t.join()

cv2.destroyAllWindows()

 