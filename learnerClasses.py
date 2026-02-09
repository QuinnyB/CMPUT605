import math
import numpy as np

class TDLearner:
    def __init__(self, alpha, gamma, feature_vector_length):
        self.alpha = alpha
        self.gamma = gamma
        self.w = np.zeros(feature_vector_length)
        self.x_cur = np.zeros_like(self.w)
        self.verifier_buffer_length = math.ceil(5*(1/(1-gamma)))  # Number of steps to look back at for verification
        self.verifier_indices = np.arange(self.verifier_buffer_length)    # Create an array of indices [0, 1, 2, ..., verifier_buffer_length-1]

    def update(self, x_next, c_next):
        # Calculate TD error: delta = c_next + gamma * w*x_next - w*x_cur
        delta = c_next + self.gamma * (self.w @ x_next) - (self.w @ self.x_cur)
        # Update weights: w = w + alpha * delta * x_cur
        self.w += self.alpha * delta * self.x_cur
        # Update x_cur 
        self.x_cur = x_next
        return delta

    def get_prediction(self, x):
        # pred = w*x * (1 - gamma)
        return (self.w @ x) * (1 - self.gamma)