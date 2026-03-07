''' 
Main code for CMPUT 605 Robot Module 4 - Objectibve 2: Discrete Actor-Critic Learning
Written by: Quinn Boser, with assistance from Google Gemini 
Feb. 2026
'''

import math
import random
import threading
from pynput import keyboard
from RM4_robotClass import MiniBento
from RM4_learnerClass import ACLearner
from RM4_visualizerClass import ACVisualizer
from RM4_helperFunctions import *

# --- Configuration -----------------------------------------------------------------------------
# Robot arm:
COMM_PORT = 'COM15'     # Lab Mini Bento likes port 13, home likes 15
BAUDRATE = 1000000
MOTOR_VELO = 20
INITIAL_POSITIONS = {1: 2048, 2: 1800, 4: 2700, 5: 2780}
MOTOR_ID = 1
MOTOR_LIM_1 = 1500
MOTOR_LIM_2 = 2500

# Learning:
NUM_POS_BINS = 10  # For creating feature vector
MOVE_AMOUNT = math.floor((MOTOR_LIM_2 - MOTOR_LIM_1)/NUM_POS_BINS) # Amount to move when taking an action (in motor encoder units)
WAIT_TIME = (MOVE_AMOUNT / (4096 * 0.229 * MOTOR_VELO)) * 60
GOAL_POS = random.randint(MOTOR_LIM_1, MOTOR_LIM_2)
agent_params = {
    "actor_alpha": 0.5,
    "critic_alpha": 0.8,
    "avg_reward_alpha": 0.1,
    "num_actions": 3,   # Stay [0], move down [1], or move up [2] (in motor coord system)
    "feature_vector_length": NUM_POS_BINS
    }

# Misc:
loop_count = 0
is_paused = False   # Global pause flag
running = True      # Control flag to stop threads

# Function pointers so the robot class can check global flags
def get_paused(): return is_paused
def get_running(): return running

# --- Set up robot, learner, and visualizer  -------------------------------------------------------
with MiniBento(COMM_PORT, BAUDRATE, MOTOR_VELO, INITIAL_POSITIONS) as arm:
    learner = ACLearner(agent_params)
    plotter = ACVisualizer(window_size=100, goal_pos=GOAL_POS)

    # Define keyboard event handler
    def on_press(key):
        global is_paused
        # Spacebar to Pause/Resume
        if key == keyboard.Key.space:
            is_paused = not is_paused
            print(f"*** {'PAUSED' if is_paused else 'RESUMED'} ***")
            if is_paused:
                # Stop motor where it is
                p, _, _ = arm.read_from_motor(MOTOR_ID)
                arm.set_goal_pos(MOTOR_ID, p)

    # Start the Keyboard Listener Thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    time.sleep(0.5)  # Delay to let arm get into initial position
# -----------------------------------------------------------------------------------------
# --- Main Loop  --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
    while plotter.is_open():
        if is_paused:
            plotter.process_events() # Handle window events
            continue
        
        # Increment loop count
        loop_count += 1

        # Get next state from the robot
        pos, _, _ = arm.read_from_motor(MOTOR_ID)
        if pos is None: continue

        # Convert position into feature vector X
        x_next, pos_bin = featurize_pos(pos, MOTOR_LIM_1, MOTOR_LIM_2, NUM_POS_BINS)

        # Calculate reward (negative distance from goal position)
        reward_next = -10*abs(pos - GOAL_POS)/(MOTOR_LIM_2 - MOTOR_LIM_1)  # Normalize reward to be between -1 and 0

        print(f"Loop {loop_count}: Position={pos}, Position Bin={pos_bin}, Reward={reward_next}, Avg Reward={learner.avg_reward}")

        # Visualize
        plotter.update_data(pos, reward_next) 
        plotter.draw()

        # Update learner with new state and reward
        if learner.last_action is not None:  # Skip update on first loop since we don't have a previous action yet
            learner.update(reward_next, x_next)

        # Get next action from learner and take action on robot
        action = learner.get_next_action()
        if action == 0:   # Stay
            pass
        elif action == 1: # Move down
            arm.move_x_amount(MOTOR_ID, -MOVE_AMOUNT, min_pos=MOTOR_LIM_1, max_pos=MOTOR_LIM_2)
        elif action == 2: # Move up
            arm.move_x_amount(MOTOR_ID, MOVE_AMOUNT, min_pos=MOTOR_LIM_1, max_pos=MOTOR_LIM_2)

        # time.sleep(WAIT_TIME + 0.01)  # Wait to get to position before next loop iteration
        time.sleep(1)
        print("\n")
    
    running = False