import numpy as np
from filterpy.kalman import KalmanFilter
from .detection import Detection

class Tracklet():
    def __init__(self, id, bbox, min_hits=3):
        self.id = id
        
        self.min_hits = min_hits
        self.hit_streak = 1  # Number of consecutive detections for the tracker
        self.time_since_update = 0
        self.age = 1
        self.in_probation = True
        self.history = []

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = bbox.to_xysr().reshape((4,1))


    def predict(self):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.age += 1

        # Keep aspect ratio positive
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        
        self.kf.predict()

    def update(self, detection):
        self.hit_streak += 1
        self.time_since_update = 0
        if self.hit_streak >= self.min_hits:
            self.in_probation = False
        
        self.kf.update(detection.to_xysr().reshape((4,1)))

    def get_state(self):
        x = self.kf.x
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        
        return Detection([x[0]-w/2., x[1]-h/2., w, h], 1.0)

    def record_state(self):
        state = self.get_state()
        self.history.append(state)
