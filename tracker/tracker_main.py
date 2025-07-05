import numpy as np
from scipy.optimize import linear_sum_assignment
from .track import Track

def iou(bb1, bb2):
    xA = max(bb1[0], bb2[0])
    yA = max(bb1[1], bb2[1])
    xB = min(bb1[2], bb2[2])
    yB = min(bb1[3], bb2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    boxBArea = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

class Sort:
    def __init__(self, max_age=5, iou_threshold=0.3):
        """
        max_age: quanti frame tenere vivi i track senza associazioni
        iou_threshold: soglia minima di IoU per associare una detection a un track
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.track_id_count = 0

    def update(self, detections):
        # 1) Predict
        for track in self.tracks:
            track.predict()

        # 2) Association (Hungarian)
        if len(self.tracks) == 0:
            matched, unmatched_dets = [], list(range(len(detections)))
        else:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for t, track in enumerate(self.tracks):
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = iou(track.bbox, det)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched, unmatched_dets = [], list(range(len(detections)))
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] < self.iou_threshold:
                    continue
                matched.append((r, c))
                unmatched_dets.remove(c)

        # 3) Update matched tracks
        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(detections[d_idx])

        # 4) Create new tracks per ogni detection non associata
        for idx in unmatched_dets:
            self.tracks.append(Track(detections[idx], self.track_id_count))
            self.track_id_count += 1

        # 5) Rimuovi i track troppo “vecchi”
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Ritorna lista di (id, bbox)
        return [(t.id, t.bbox) for t in self.tracks]
