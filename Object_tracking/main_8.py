from ultralytics import YOLO
import threading
import cv2
import time
import numpy as np

# Settings
NUM_STREAMS = 24
GRID_COLS = 6
GRID_ROWS = 4
FRAME_WIDTH = 213
FRAME_HEIGHT = 160
FPS_LIMIT = 1  # 1 frame per second

# Sample model and video — duplicated for demo
VIDEO_PATH = "../Resources/Videos/video1.mp4"
MODEL_NAME = "yolo11n.pt"

# Repeat sources for demo purposes
VIDEO_PATHS = [VIDEO_PATH for _ in range(NUM_STREAMS)]
MODEL_NAMES = [MODEL_NAME for _ in range(NUM_STREAMS)]

# Shared state
frames = [None for _ in range(NUM_STREAMS)]
fps_values = [0.0 for _ in range(NUM_STREAMS)]
locks = [threading.Lock() for _ in range(NUM_STREAMS)]


def run_tracker(index, model_name, source):
    model = YOLO(model_name)
    prev_process_time = time.time()

    for result in model.track(source=source, stream=True, show=False):
        current_time = time.time()
        elapsed = current_time - prev_process_time

        if elapsed < 1.0 / FPS_LIMIT:
            continue

        frame = result.plot()
        if frame is not None:
            prev_process_time = current_time
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            fps = 1 / elapsed
            cv2.putText(frame, f"FPS: {fps:.2f}", (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            with locks[index]:
                frames[index] = frame
                fps_values[index] = fps


# Start threads
for i in range(NUM_STREAMS):
    t = threading.Thread(target=run_tracker, args=(i, MODEL_NAMES[i], VIDEO_PATHS[i]), daemon=True)
    t.start()

# Display loop
while True:
    current_frames = []

    for i in range(NUM_STREAMS):
        with locks[i]:
            frame = frames[i]

        if frame is None:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, f"Waiting {i+1}", (30, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        current_frames.append(frame)

    # Create grid rows
    rows = []
    for i in range(0, NUM_STREAMS, GRID_COLS):
        row = np.hstack(current_frames[i:i+GRID_COLS])
        rows.append(row)

    combined = np.vstack(rows)
    cv2.imshow("24 Stream YOLO11 Tracker (6x4 Grid)", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
