from ultralytics import YOLO

def main():
    # 1) carica modello small (più adatto a dataset limitati)
    model = YOLO('yolov8s.pt')
    
    # 2) parametri di training avanzati
    results = model.train(
        data='data/ball_detection.yaml',
        epochs=100,                       # numero di epoche
        batch=16,                         # batch size
        imgsz=640,                        # risoluzione di training
        freeze=[0, 10],                   # sblocca i primi 10 layer del backbone
        lr0=5e-5,                         # learning rate iniziale più basso
        lrf=0.01,                         # fattore di fine LR
        cos_lr=True,                      # decay coseno
        mosaic=1.0,                       # mosaic augmentation
        copy_paste=0.5,                   # copy-paste augmentation
        name='ball_detector_ft'           # nome della run
    )

    # 3) validazione finale
    val_results = model.val(
        data='data/ball_detection.yaml',
        imgsz=640,
        batch=16,
        name='ball_detector_ft'
    )

    print('Risultati di validazione:', val_results)

if __name__ == '__main__':
    main()
