import math
import numpy as np
from collections import deque

class TDLearner:
    def __init__(self, alpha, default_gamma, feature_vector_length, history_length):
        self.alpha = alpha
        self.gamma = default_gamma
        self.w = np.zeros(feature_vector_length)
        self.x_cur = np.zeros(feature_vector_length, dtype=int) 
        self.h_len = history_length
        self.gamma_history = deque([default_gamma] * self.h_len, maxlen=self.h_len)
        self.c_history = deque([0.0] * self.h_len, maxlen=self.h_len)
        self.pred_history = deque([0.0 * (self.h_len+1)], maxlen=self.h_len+1)  # Make prediction buffer one bigger 
    
    def update(self, x_next, c_next, gamma_next=None):
        # If gamma is not provided, use the default gamma
        if gamma_next is None: gamma_next = self.gamma
        # Calculate TD error: delta = c_next + gamma * w*x_next - w*x_cur
        delta = c_next + gamma_next * (self.w @ x_next) - (self.w @ self.x_cur)
        # Update weights: w = w + alpha * delta * x_cur
        self.w += self.alpha * delta * self.x_cur
        # Update x_cur 
        self.x_cur = x_next
        # Calculate next prediction 
        pred = self.w @ self.x_cur
        # Store history 
        self.gamma_history.append(gamma_next)
        self.c_history.append(c_next)
        self.pred_history.append(pred) # To do - check if this correct, or should be stored one easrlier
        return pred

    def get_prediction(self, x):
        return self.w @ x
    
    def get_verifier_data(self):
        # Convert deques to lists for math
        gamma_hist = list(self.gamma_history)
        c_hist = list(self.c_history)
        # Create an array of indices [0, 1, 2, ..., learner.h_len-1]
        verifier_indices = np.arange(self.h_len)    
        # Compute expected prediction for learner.h_len steps in the past
        expected_pred = np.sum(c_hist * (gamma_hist ** verifier_indices))
        # To do - add error ?
        return expected_pred, self.h_len # Need to figure out if this should be h_len or h_len+1 