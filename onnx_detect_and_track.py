import cv2
import numpy as np
import onnxruntime as ort
from tracker.tracker_main import Sort
from tqdm import tqdm

ort.set_default_logger_severity(3) # Suppress ONNX Runtime warnings

# Load ONNX model using DirectML
session = ort.InferenceSession("yolov8s.onnx", providers=["DmlExecutionProvider"])
input_name = session.get_inputs()[0].name

# Initialize trackers
player_tracker = Sort()
ball_tracker = Sort()

# Open input video
video_path = "input/2.mp4"
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video setup
out = cv2.VideoWriter("output/tracked_2.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps,
                      (frame_width, frame_height))

# Progress bar
progress = tqdm(total=total_frames, desc="Processing ONNX", unit="frame", dynamic_ncols=True, leave=True)

def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = preprocess(frame)
    outputs = session.run(None, {input_name: input_tensor})
    detections = outputs[0][0]  # Shape: (N, 84)

    input_w, input_h = 640, 640
    x_scale = frame.shape[1] / input_w
    y_scale = frame.shape[0] / input_h

    player_detections = []
    ball_detections = []

    for det in detections:
        x1, y1, x2, y2 = det[:4]
        scores = det[4:]
        conf = np.max(scores)
        cls = np.argmax(scores)

        if conf < 0.3:
            continue

        # Rescale to original frame size
        x1 *= x_scale
        y1 *= y_scale
        x2 *= x_scale
        y2 *= y_scale

        if int(cls) == 0:  # Person
            player_detections.append([x1, y1, x2, y2])
        elif int(cls) == 32:  # Sports ball
            ball_detections.append([x1, y1, x2, y2])

    tracked_players = player_tracker.update(player_detections)
    tracked_balls = ball_tracker.update(ball_detections)

    # Draw tracked players (green)
    for track_id, bbox in tracked_players:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Player {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw tracked balls (light blue)
    for track_id, bbox in tracked_balls:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 100), 2)
        cv2.putText(frame, f"Ball {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)

    out.write(frame)
    progress.update(1)

# Cleanup
progress.close()
cap.release()
out.release()
cv2.destroyAllWindows()
print("\nVideo processing complete with ONNX! Output saved.")
