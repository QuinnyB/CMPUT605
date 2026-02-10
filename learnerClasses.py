import math
import numpy as np
from collections import deque

class TDLearner:
    def __init__(self, alpha, default_gamma, feature_vector_length, history_length):
        self.alpha = alpha
        self.gamma = default_gamma
        self.w = np.zeros(feature_vector_length)
        self.x_cur = np.zeros_like(self.w)
        self.h_len = history_length
        self.gamma_history = deque([default_gamma] * self.h_len, maxlen=self.h_len)
        self.c_history = deque([0.0] * self.h_len, maxlen=self.h_len)
    
    def update(self, x_next, c_next, gamma_next=None):
        # If gamma is not provided, use the default gamma
        if gamma_next is None: gamma_next = self.gamma
        # Calculate TD error: delta = c_next + gamma * w*x_next - w*x_cur
        delta = c_next + gamma_next * (self.w @ x_next) - (self.w @ self.x_cur)
        # Update weights: w = w + alpha * delta * x_cur
        self.w += self.alpha * delta * self.x_cur
        # Update x_cur 
        self.x_cur = x_next
        # Store history (Circular Buffer)
        self.gamma_history.append(gamma_next)
        self.c_history.append(c_next)
        return delta

    def get_prediction(self, x):
        # pred = w*x * (1 - gamma)
        return (self.w @ x) * (1 - self.gamma)
    
    