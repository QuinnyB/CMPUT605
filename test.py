
import math
MOTOR_LIM_1 = 1500
MOTOR_LIM_2 = 2500

# Learning:
NUM_POS_BINS = 10  # For creating feature vector
MOVE_AMOUNT = math.floor((MOTOR_LIM_2 - MOTOR_LIM_1)/NUM_POS_BINS)
print(MOVE_AMOUNT)