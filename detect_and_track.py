import cv2
import numpy as np
from ultralytics import YOLO
from tracker.tracker_main import Sort
from tqdm import tqdm
import os

def main():
    # 1. Carica YOLOv8x (scarica yolov8x.pt nella cartella se non l'hai già)
    model = YOLO("yolov8x.pt")

    # 2. Inizializza i tracker SORT
    player_tracker = Sort()
    ball_tracker   = Sort()

    # 3. Apri il video di input
    video_path   = "input/2.mp4"
    cap          = cv2.VideoCapture(video_path)
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. Prepara cartella di output e VideoWriter
    os.makedirs("output", exist_ok=True)
    output_path = "output/tracked_2_yolox.mp4"
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h)
    )

    # 5. Barra di avanzamento
    progress = tqdm(
        total=total_frames,
        desc="Processing YOLOv8x",
        unit="frame",
        dynamic_ncols=True,
        leave=True
    )

    # 6. Per ogni frame: rilevamento + tracking + disegno
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 6.1 inference
        results = model(frame)[0]

        player_detections = []
        ball_detections   = []

        # 6.2 estrai bbox da YOLOv8x
        for det in results.boxes:
            cls  = int(det.cls[0])           # indice di classe
            conf = float(det.conf[0])        # confidenza
            x1, y1, x2, y2 = det.xyxy[0].tolist()

            if cls == 0 and conf > 0.4:      # persona
                player_detections.append([x1, y1, x2, y2])
            elif cls == 32 and conf > 0.2:   # palla da sport
                ball_detections.append([x1, y1, x2, y2])

        # 6.3 aggiorna trackers
        tracked_players = player_tracker.update(player_detections)
        tracked_balls   = ball_tracker.update(ball_detections)

        # 6.4 disegna i box verdi per i giocatori
        for track_id, bbox in tracked_players:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                f"Player {track_id}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        # 6.5 disegna i box azzurri per la palla
        for track_id, bbox in tracked_balls:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255,200,100), 2)
            cv2.putText(
                frame,
                f"Ball {track_id}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,200,100),
                2
            )

        # 6.6 scrivi il frame elaborato e aggiorna barra
        out.write(frame)
        progress.update(1)

    # 7. Cleanup
    progress.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nVideo processing complete with YOLOv8x! Saved to: {output_path}")

if __name__ == "__main__":
    main()
