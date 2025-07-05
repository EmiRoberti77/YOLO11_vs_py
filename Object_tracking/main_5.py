from ultralytics import YOLO
import threading
import cv2
import time
import numpy as np

video_path_a = "../Resources/Videos/video7.mp4"
video_path_b = "../Resources/Videos/video8.mp4"
MODEL_NAMES = ["yolo11n.pt", "yolo11n-seg.pt"]
SOURCES = [video_path_a, video_path_b]

# Shared state
frames = [None, None]
fps_values = [0.0, 0.0]
locks = [threading.Lock(), threading.Lock()]

def run_tracker(index, model_name, source):
    model = YOLO(model_name)
    prev_time = time.time()

    for result in model.track(source=source, stream=True, show=False):
        frame = result.plot()

        if frame is not None:
            frame = cv2.resize(frame, (640, 360))

            # FPS calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Draw FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Update shared state with lock
            with locks[index]:
                frames[index] = frame
                fps_values[index] = fps

# Launch threads
for i, (model_name, source) in enumerate(zip(MODEL_NAMES, SOURCES)):
    t = threading.Thread(target=run_tracker, args=(i, model_name, source), daemon=True)
    t.start()

# Main loop
while True:
    current_frames = []

    for i in range(2):
        with locks[i]:
            frame = frames[i]
        
        if frame is None:
            # Create a dummy "waiting" frame
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(frame, f"Waiting for Stream {i+1}", (50, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        current_frames.append(frame)

    combined = np.hstack(current_frames)
    cv2.imshow("Combined Streams", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
