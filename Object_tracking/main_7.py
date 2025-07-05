from ultralytics import YOLO
import threading
import cv2
import time
import numpy as np

# Video paths and models
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
FPS_LIMIT = 1  # 1 frame per second

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

        # Skip until 1 second has passed
        if elapsed < 1.0 / FPS_LIMIT:
            continue

        frame = result.plot()

        if frame is not None:
            prev_process_time = current_time

            # Resize
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # FPS calculation
            fps = 1 / elapsed
            fps_values[index] = fps

            # Overlay FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            with locks[index]:
                frames[index] = frame


# Launch threads
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
            cv2.putText(frame, f"Waiting {i+1}", (50, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        current_frames.append(frame)

    # 3x2 grid layout
    row1 = np.hstack(current_frames[:3])
    row2 = np.hstack(current_frames[3:])
    combined = np.vstack([row1, row2])

    cv2.imshow("6 Stream YOLO11 Tracker (1 FPS)", combined)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
