from ultralytics import YOLO

def main():
    # 1) carica modello small (più adatto a dataset limitati)
    model = YOLO('yolov8s.pt')
    
    # 2) parametri di training
    results = model.train(
        data='data/ball_detection.yaml',
        epochs=100,                       # più epoche
        batch=16,                         # batch size medio
        imgsz=640,                        # risoluzione standard
        freeze=[0],                       # congela il backbone (layer 0)
        lr0=1e-4,                         # learning rate ridotto
        mosaic=1.0,                       # abilita mosaic
        copy_paste=0.5,                   # abilita copy-paste augmentation
        name='ball_detector_small'        # nome della run
    )

    # 3) fai validazione finale
    val_results = model.val(
        data='data/ball_detection.yaml',
        imgsz=640,
        batch=16,
        name='ball_detector_small'
    )

    print('Risultati di validazione:', val_results)

if __name__ == '__main__':
    main()
