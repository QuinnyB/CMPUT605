''' 
Main code for CMPUT 605 Robot Module 4 - Objectibve 2: Discrete Actor-Critic Learning
Written by: Quinn Boser, with assistance from Google Gemini 
Feb. 2026
'''

import math
import threading
from pynput import keyboard
from RM4_robotClass import MiniBento
from RM4_learnerClass import TDLearner
from RM4_helperFunctions import *

# --- Configuration -----------------------------------------------------------------------------
# Robot arm:
COMM_PORT = 'COM15'     # Lab Mini Bento likes port 13, home likes 15
BAUDRATE = 1000000
MOTOR_VELO = 20
INITIAL_POSITIONS = {1: 2048, 2: 1700, 4: 2600, 5: 2780}
MOTOR_ID = 4
MOTOR_LIM_1= 2150
MOTOR_LIM_2 = 3200

# Learning:
NUM_POS_BINS = 10   # For creating feature vector
agent_params = {
    "actor_alpha": [0.1],
    "critic_alpha": [0.1],
    "avg_reward_alpha": [0.1],
    "num_actions": 3,
    "feature_vector_length": [NUM_POS_BINS],
}

