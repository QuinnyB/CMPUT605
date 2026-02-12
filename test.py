import numpy as np
from collections import deque

c_hist = np.array([1, 1, 1, 1])
gamma_hist = np.array([0.5, 0.5, 0, 0])

discounts = np.concatenate((np.array([1]), np.cumprod(gamma_hist[:-1])))
print(discounts)
expected_pred = np.sum(c_hist * discounts)
print(expected_pred)

