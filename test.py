import numpy as np
from collections import deque
test = deque([np.nan] * 10, maxlen=10)
print(test)
test[-3] = 1
print(test)
test.append(np.nan)
print(test)
