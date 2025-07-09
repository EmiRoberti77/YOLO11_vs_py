from utils.video_utils import (read_video, save_video)
from detections import CarDetection

__MAIN__ = "__main__"

def main():
  # Set input and output
  input_video_path = "../Resources/Videos/car4.mp4"
  output_video_path = "output_videos/output.avi"

  
  # extract video frames
  video_frames = read_video(input_video_path)
  
  # Detect Cars
  car_detector = CarDetection(model_path="yolo11n.pt")
  car_detections = car_detector.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/car_detection.pkl")
  
  # Draw the car binding boxes
  car_detector.draw_bboxes(video_frames, car_detections)

  # Save frames into video output
  save_video(video_frames,output_video_path)


if __name__ == __MAIN__:
  main()