import numpy as np
from collections import deque
import math

gamma_history = deque([0, 3, 5, 2])
print(gamma_history[-1])