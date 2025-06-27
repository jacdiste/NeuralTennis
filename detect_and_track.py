import cv2
from ultralytics import YOLO
from tracker.tracker_main import Sort

# Load YOLOv8 model
model = YOLO("yolov8n.pt")
tracker = Sort()

# Open video
cap = cv2.VideoCapture("input/1.mp4")

# Get video properties for saving
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
out = cv2.VideoWriter("output/tracked_1.mp4", fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 detection
    results = model(frame)[0]
    detections = []

    for det in results.boxes:
        cls = int(det.cls[0])
        if cls in [0, 32]:  # person or sports ball
            x1, y1, x2, y2 = map(float, det.xyxy[0])
            detections.append([x1, y1, x2, y2])

    # Tracking with SORT
    tracked = tracker.update(detections)

    # Draw detections and tracking boxes
    for tid, bbox in tracked:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

    cv2.imshow("YOLOv8 + SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
