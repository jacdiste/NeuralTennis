import os
import cv2
import torch
from ultralytics import YOLO
from tracker.tracker_main import Sort
from tqdm import tqdm

# --- CONFIG -------------------------
INPUT_VIDEO  = "input/1.mp4"
OUTPUT_VIDEO = "output/tracked_1.mp4"
YOLO_WEIGHTS = "yolov8x.pt"
PLAYER_CONF  = 0.4
BALL_CONF    = 0.2
# ------------------------------------

def main():
    # 1) device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2) load model (sul device)
    model = YOLO(YOLO_WEIGHTS)
    model.model.to(device)

    # 3) init trackers
    player_tracker = Sort()
    ball_tracker   = Sort()

    # 4) apri video e writer
    cap   = cv2.VideoCapture(INPUT_VIDEO)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_VIDEO,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))

    # 5) barra di progresso on one line
    pbar = tqdm(total=total,
                desc="Processing",
                unit="frame",
                dynamic_ncols=True,
                leave=False)  # leave=False evita che la barra resti sul terminale a fine run

    # 6) loop sui frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 7) inference YOLO con logging disabilitato
        results = model(frame,
                        device=device,
                        imgsz=1600,
                        verbose=False,  # disabilita il print dei risultati YOLO
                        show=False      # disabilita l’apertura della finestra di preview
                        )[0]
        dets = results.boxes  # contiene xyxy, conf, cls

        # 8) separa detections
        player_dets, ball_dets = [], []
        for box in dets:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if cls == 0 and conf >= PLAYER_CONF:
                player_dets.append([x1, y1, x2, y2])
            elif cls == 32 and conf >= BALL_CONF:
                ball_dets.append([x1, y1, x2, y2])

        # 9) aggiorna tracker
        tracks_p = player_tracker.update(player_dets)
        tracks_b = ball_tracker.update(ball_dets)

        # 10) disegna players
        for tid, bbox in tracks_p:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"P{tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 11) disegna balls
        for tid, bbox in tracks_b:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 100), 2)
            cv2.putText(frame, f"B{tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

        # 12) scrivi e aggiorna progress bar
        out.write(frame)
        pbar.update(1)

    # 13) cleanup
    pbar.close()
    cap.release()
    out.release()
    print(f"\nVideo saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
