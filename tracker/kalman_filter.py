import numpy as np

class KalmanFilter:
    def __init__(self):
        # Stato: [x, y, dx, dy]
        self.dt = 1.0
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.eye(2, 4)  # Osserviamo solo x, y
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 10.0
        self.P = np.eye(4) * 500.
        self.x = np.zeros((4, 1))  # stato iniziale

    def initiate(self, cx, cy):
        self.x[:2] = np.array([[cx], [cy]])

    def predict(self):
        self.x = self.A @ self.x 
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x[:2].flatten()

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x += K @ y 
        self.P = (np.eye(4) - K @ self.H) @ self.P
