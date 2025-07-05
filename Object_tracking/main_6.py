from ultralytics import YOLO
import threading
import cv2
import time
import numpy as np

# 6 video sources and model names
VIDEO_PATHS = [
    "../Resources/Videos/video1.mp4",
    "../Resources/Videos/video2.mp4",
    "../Resources/Videos/video3.mp4",
    "../Resources/Videos/video4.mp4",
    "../Resources/Videos/video5.mp4",
    "../Resources/Videos/video6.mp4"
]

MODEL_NAMES = [
    "yolo11n.pt",
    "yolo11n.pt",
    "yolo11n.pt",
    "yolo11n.pt",
    "yolo11n.pt",
    "yolo11n.pt"
]

NUM_STREAMS = 6
FRAME_WIDTH, FRAME_HEIGHT = 320, 240

# Shared frame buffers and locks
frames = [None for _ in range(NUM_STREAMS)]
fps_values = [0.0 for _ in range(NUM_STREAMS)]
locks = [threading.Lock() for _ in range(NUM_STREAMS)]


def run_tracker(index, model_name, source):
    model = YOLO(model_name)
    prev_time = time.time()

    for result in model.track(source=source, stream=True, show=False):
        frame = result.plot()

        if frame is not None:
            # Resize for layout
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # FPS calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            # Draw FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Store frame safely
            with locks[index]:
                frames[index] = frame
                fps_values[index] = fps


# Start a thread per stream
for i in range(NUM_STREAMS):
    t = threading.Thread(target=run_tracker, args=(i, MODEL_NAMES[i], VIDEO_PATHS[i]), daemon=True)
    t.start()

# Main GUI display loop
while True:
    current_frames = []

    for i in range(NUM_STREAMS):
        with locks[i]:
            frame = frames[i]

        if frame is None:
            # Create placeholder
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, f"Waiting {i+1}", (50, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        current_frames.append(frame)

    # Combine into 3x2 grid
    row1 = np.hstack(current_frames[:3])
    row2 = np.hstack(current_frames[3:])
    combined = np.vstack([row1, row2])

    cv2.imshow("6 Stream YOLO11 Tracker", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
