import numpy as np

class Sort:
    def __init__(self):
        self.trackers = []
        self.track_id = 0

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            track = [x1, y1, x2, y2, self.track_id]
            self.track_id += 1
            updated_tracks.append(track)
        return np.array(updated_tracks)
