'''
TD Learner class for CMPUT 605 Robot Module 2
Written by: Quinn Boser, with assistance from Google Gemini 
Feb. 2026
'''

import math
import numpy as np
from collections import deque

class TDLearner:
    def __init__(self, alpha, default_gamma, feature_vector_length, history_length, initial_w=None, lambda_=None):
        self.alpha = alpha
        self.gamma = default_gamma
        self.x_cur = np.zeros(feature_vector_length, dtype=int) 
        if initial_w is None:
            self.w = np.zeros(feature_vector_length) 
        else: 
            self.w = np.full(feature_vector_length, initial_w)
        self.h_len = history_length
        self.gamma_history = deque([default_gamma] * self.h_len, maxlen=self.h_len)
        self.c_history = deque([0.0] * self.h_len, maxlen=self.h_len)
        # For eligibility traces version of update:
        if lambda_ is not None:
            self.lambda_ = lambda_
            self.e = np.zeros(feature_vector_length)   
    
    # Update weigts using TD error forumla and return prediction for input state
    def update(self, x_next, c_next, gamma_next=None, importance_ratio=1.0):
        # If gamma is not provided, use the default gamma
        if gamma_next is None: gamma_next = self.gamma
        # Calculate TD error: delta = c_next + gamma * w*x_next - w*x_cur
        delta = c_next + gamma_next * (self.w @ x_next) - (self.w @ self.x_cur)
        # print(f"Weight of x_cur: {self.w @ self.x_cur}, Weight of x_next: {self.w @ x_next}")
        # print(f"TD Error (delta): {delta}")
        # Update weights: w = w + alpha * delta * x_cur
        self.w += self.alpha * delta * self.x_cur * importance_ratio
        # Calculate prediction
        pred = self.w @ x_next
        # Update x_cur 
        self.x_cur = x_next
        # Store history 
        self.gamma_history.append(gamma_next)
        self.c_history.append(c_next)
        return pred
    
    # Update weights using TD error and eligbiity traces and return prediction for input state
    def update_withEligibilityTraces(self, x_next, c_next, gamma_next=None, importance_ratio=1.0):
        # If gamma is not provided, use the default gamma
        if gamma_next is None: 
            gamma_next = self.gamma
        gamma_cur = self.gamma_history[-1]
        # Calculate TD error: delta = c_next + gamma_next * w*x_next - w*x_cur
        delta = c_next + gamma_next * (self.w @ x_next) - (self.w @ self.x_cur)
        # Check that self.e exists (i.e. that lambda was set in init)
        if not hasattr(self, 'e'):
            raise AttributeError("Eligibility traces not initialized. Set lambda_ in constructor.")
        # Update eligibility traces: e = gamma_cur*lambda*e + x_cur
        # Compute element-wise minimum between e and 1 to prevent explosion of traces (replacing traces))
        self.e = np.minimum(gamma_cur*self.lambda_*self.e + self.x_cur, 1.0)
        # Update weights: w = w + alpha * delta * e
        self.w += self.alpha*delta*self.e*importance_ratio
        # Calculate next prediction 
        pred = self.w @ x_next
        # Update x_cur 
        self.x_cur = x_next
        # Store history 
        self.gamma_history.append(gamma_next)
        self.c_history.append(c_next)
        return pred

    # Get learner prediction of target for input state
    def get_prediction(self, x):
        return self.w @ x
    
    # Used stored data to calculate idealized prediction for self.h_len steps in the past
    def get_verifier_data(self):
        # Convert deques to lists for math
        gamma_hist = list(self.gamma_history)
        c_hist = list(self.c_history)
        # Compute expected prediction for learner.h_len steps in the past
        # G = C0 + yC1 + (y^2)C2 + (y^3)C3 + ... (for fixed gamma)
        # G = C0 + y0C1 + y0y1C2 + y0y1y2C3 + ... (for variable gamma)
        discounts = np.concatenate((np.array([1]), np.cumprod(gamma_hist[:-1])))
        expected_pred = np.sum(c_hist * discounts)
        return expected_pred, self.h_len
    
    # Clear 
    def clear_eligibility_traces(self):
        if not hasattr(self, 'e'):
            raise AttributeError("Eligibility traces not initialized. Set lambda_ in constructor.")
        self.e.fill(0)
 
