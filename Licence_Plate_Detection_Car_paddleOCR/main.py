from utils.video_utils import (read_video, save_video)

def main():
  input_video_path = "../Resources/Videos/car4.mp4"
  output_video_path = "output_videos/output.avi"
  video_frames = read_video(input_video_path)
  save_video(video_frames,output_video_path)


if __name__ == "__main__":
  main()