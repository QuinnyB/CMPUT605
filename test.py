import numpy as np
from collections import deque
import math

MOTOR_VELO = 20
MOTOR_LIM_1= 2150
MOTOR_LIM_2 = 3200
MOTOR_RANGE = MOTOR_LIM_2 - MOTOR_LIM_1
WAIT_TIME = ((MOTOR_RANGE) / (4096 * 0.229 * MOTOR_VELO)) * 60

print(WAIT_TIME)
