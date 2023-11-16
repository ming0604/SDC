import numpy as np
from math import cos, sin, atan2

class ExtendedKalmanFilter:
    def __init__(self,x=0, y=0, yaw=0):
        # Define what state to be estimate
        # Ex.
        #   only pose -> np.array([x, y, yaw])
        #   with velocity -> np.array([x, y, yaw, vx, vy, vyaw])
        #   etc...
        self.pose = np.array([x, y, yaw]) #???
        
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        
        # State covariance matrix
        self.S = np.identity(3) * 1
        
        # Observation matrix
        self.C = np.array([[1,0,0],[0,1,0]])
        #self.C = np.identity(3)
        
        # State transition error
        self.R = np.identity(3) * 1
        
        # Measurement error
        self.Q = np.identity(3) * 1
        print("Initialize Kalman Filter")
    
    def predict(self, u):
        # Base on the Kalman Filter design in Assignment 3
        # Implement a linear or nonlinear motion model for the control input
        # Calculate Jacobian matrix of the model as self.A

        #use transition matrix of world and last car local frame to get the nonlinear motion model
        self.u_transition_matrix = np.array([ [cos(self.pose[2]), -sin(self.pose[2]), 0],
                                            [sin(self.pose[2]), cos(self.pose[2]), 0],
                                            [0,0,1] ])
        #A is Jacobian matrix here
        self.A = np.array([ [1, 0, (-u[0]*sin(self.pose[2])-u[1]*cos(self.pose[2]))],
                            [0, 1, (u[0]*cos(self.pose[2])-u[1]*sin(self.pose[2]))],
                            [0, 0, 1] ])

        self.pose = self.pose + np.dot(self.u_transition_matrix, u) 
        self.S = np.dot(np.dot(self.A, self.S) , self.A.T) + self.R
        
        #raise NotImplementedError
    
        
    def update(self, z):
        # Base on the Kalman Filter design in Assignment 3
        # Implement a linear or nonlinear observation matrix for the measurement input
        # Calculate Jacobian matrix of the matrix as self.C
        
        #use linear observation model as hw3 
        K = np.dot(np.dot(self.S, self.C.T), np.linalg.inv((np.dot(np.dot(self.C, self.S), self.C.T) + self.Q)))
        self.pose = self.pose + np.dot(K, (z - np.dot(self.C, self.pose)) )
        self.S = np.dot((np.identity(3)-np.dot(K, self.C)), self.S)
        return self.pose, self.S

        #raise NotImplementedError
        
    
    
    
        