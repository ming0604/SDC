import numpy as np

class KalmanFilter:
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.state = np.array([x, y, yaw])
        
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        
        # State covariance matrix
        self.S = np.identity(3) * 1
        
        # Observation matrix
        self.C = np.array([[1,0,0],[0,1,0]])
        
        # State transition error
        self.R = np.array([[1,0,0],[0,1,0],[0,0,1]])
        
        # Measurement error
        self.Q = np.array([[3,0],[0,3]])

    def predict(self, u):
        #assume that the mean of belief is state
        self.state_predict = np.dot(self.A, self.state) + np.dot(self.B, u)
        self.S_predict = np.dot(np.dot(self.A, self.S) , self.A.T) + self.R

    def update(self, z):
        K = np.dot(np.dot(self.S_predict, self.C.T), np.linalg.inv((np.dot(np.dot(self.C, self.S_predict), self.C.T) + self.Q)))
        self.state = self.state_predict + np.dot(K, (z - np.dot(self.C, self.state_predict)) )
        self.S = np.dot((np.identity(3)-np.dot(K, self.C)), self.S_predict)
        return self.state, self.S
