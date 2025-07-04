import ultralytics
import cv2
from ultralytics import YOLO
import os

video_path = 'PPE_Part2.mp4'
model = YOLO('best.pt')



results = model()

