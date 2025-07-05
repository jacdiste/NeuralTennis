import os
import cv2
import torch
from ultralytics import YOLO
from tracker.tracker_main import Sort
from collections import deque
from tqdm import tqdm
import numpy as np

# --- CONFIG ---
INPUT_VIDEO   = "input/2.mp4"
OUTPUT_VIDEO  = "output/tracked_2.mp4"
YOLO_WEIGHTS  = "yolov8x.pt"
PLAYER_CONF   = 0.4
BALL_CONF     = 0.1         # abbassato
IMG_SIZE      = 1600
MAX_AGE_BALL  = 10          # tener vivo ball-track più a lungo
IOU_THRESH_B  = 0.2         # per matching della palla
SMOOTH_WINDOW = 5           # quanti frame mediar
# ----------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) load YOLO
    model = YOLO(YOLO_WEIGHTS)
    model.model.to(device)

    # 2) init trackers
    player_tracker = Sort()  # default max_age=5, iou=0.3
    ball_tracker   = Sort(max_age=MAX_AGE_BALL, iou_threshold=IOU_THRESH_B)

    # 3) frame reader / writer
    cap = cv2.VideoCapture(INPUT_VIDEO)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_VIDEO,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w,h))

    pbar = tqdm(total=total, desc="Processing", unit="frame", dynamic_ncols=True)

    # 4) struttura per smoothing
    ball_history = {}  # tid -> deque([bbox,...])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 5) YOLO inference
        results = model(frame, device=device, imgsz=IMG_SIZE)[0]
        dets = results.boxes

        # 6) split detections
        player_dets, ball_dets = [], []
        for box in dets:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = box.xyxy[0].tolist()
            if cls == 0 and conf >= PLAYER_CONF:
                player_dets.append([x1,y1,x2,y2])
            elif cls == 32 and conf >= BALL_CONF:
                ball_dets.append([x1,y1,x2,y2])

        # 7) update trackers
        tracks_p = player_tracker.update(player_dets)
        tracks_b = ball_tracker.update(ball_dets)

        # 8) draw players
        for tid, bbox in tracks_p:
            x1,y1,x2,y2 = map(int, bbox)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"P{tid}", (x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

        # 9) smoothing + draw balls
        for tid, bbox in tracks_b:
            # init history
            if tid not in ball_history:
                ball_history[tid] = deque(maxlen=SMOOTH_WINDOW)
            ball_history[tid].append(bbox)

            # media dei bbox
            arr = np.array(ball_history[tid])
            x1, y1, x2, y2 = map(int, arr.mean(axis=0))

            cv2.rectangle(frame, (x1,y1),(x2,y2),(255,200,100),2)
            cv2.putText(frame, f"B{tid}", (x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,100),1)

        # 10) write & progress
        out.write(frame)
        pbar.update()

    pbar.close()
    cap.release()
    out.release()
    print(f"\nVideo saved to {OUTPUT_VIDEO}")

if __name__=="__main__":
    main()
