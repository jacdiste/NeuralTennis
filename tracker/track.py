from .kalman_filter import KalmanFilter

class Track:
    def __init__(self, bbox, track_id):
        cx = (bbox[0] + bbox[2]) / 2 # Calculate center of the bounding box
        cy = (bbox[1] + bbox[3]) / 2 # Calculate center of the bounding box
        self.kf = KalmanFilter()
        self.kf.initiate(cx, cy)
        self.bbox = bbox 
        self.id = track_id 
        self.time_since_update = 0 

    def predict(self):
        pred = self.kf.predict()
        self.time_since_update += 1
        return pred
    
    def update(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.kf.update([cx, cy])
        self.bbox = bbox
        self.time_since_update = 0 