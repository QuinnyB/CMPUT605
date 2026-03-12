''' 
Main code for CMPUT 605 Robot Module 4 - Objectibve 1: Discrete Actor-Critic Learning
Written by: Quinn Boser, with assistance from Google Gemini 
Feb. 2026
'''

import math
import random
from pynput import keyboard
from RM4_robotClass import MiniBento
from RM4_learnerClass import ACLearner_Discrete
from RM4_visualizerClass import ACVisualizer
from RM4_helperFunctions import *

# --- Configuration -----------------------------------------------------------------------------
# Robot arm:
COMM_PORT = 'COM13'     # Lab Mini Bento likes port 13, home likes 15
BAUDRATE = 1000000
MOTOR_VELO = 20
INITIAL_POSITIONS = {1: 2048, 2: 1800, 4: 2700, 5: 2780}
MOTOR_ID = 1
MOTOR_LIM_1 = 1500
MOTOR_LIM_2 = 2500

# Learning:
NUM_POS_BINS = 10  # For creating feature vector
MOVE_AMOUNT = math.floor((MOTOR_LIM_2 - MOTOR_LIM_1)/NUM_POS_BINS) # Amount to move when taking an action (in motor encoder units)
GOAL_POS = random.randint(MOTOR_LIM_1, MOTOR_LIM_2)
agent_params = {
    "feature_vector_length": NUM_POS_BINS,
    "num_actions": 3,   # decrease position [0], stay [1], or increase position [2]
    "avg_reward_alpha": 0.1,
    "critic_alpha": 0.9,
    "actor_alpha": 0.7
    
    # "avg_reward_alpha": 0.01,
    # "critic_alpha": 0.3,
    # "actor_alpha": 0.1,
    # "initial_avg_reward": -0.5
    }

# Plotting:
visualizer_params = {
    "window_size": 100,
    "goal_pos": GOAL_POS,
    "pos_range": [MOTOR_LIM_1, MOTOR_LIM_2],
    "reward_range": [-1.0, 0.2],
    "action_mode": 'discrete',
    "action_labels": ['Decrease Position', 'Hold Position', 'Increase Position'],
}

# Misc:
loop_count = 0
is_paused = False   # Global pause flag
running = True      # Control flag to stop threads

# Function pointers so the robot class can check global flags
def get_paused(): return is_paused
def get_running(): return running

# --- Set up robot, learner, and visualizer  ---------------------------------------------------`` ----
with MiniBento(COMM_PORT, BAUDRATE, MOTOR_VELO, INITIAL_POSITIONS) as arm:
    learner = ACLearner_Discrete(agent_params)
    plotter = ACVisualizer(visualizer_params)

    # Define keyboard event handler
    def on_press(key):
        global is_paused
        # ` key to Pause/Resume
        if key.char == '`':  
            is_paused = not is_paused
            print(f"*** {'PAUSED' if is_paused else 'RESUMED'} ***")
            if is_paused:
                # Stop motor where `it is
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
        reward_next = -abs(pos - GOAL_POS)/(MOTOR_LIM_2 - MOTOR_LIM_1)  # Normalize reward to be between -1 and 0

        # Update learner with new state and reward
        if learner.last_action is not None:  # Skip update on first loop since we don't have a previous action yet
            learner.update(reward_next, x_next)

        print(f"Loop {loop_count}: Position={pos}, Position Bin={pos_bin}, Reward={reward_next}, Avg Reward={learner.avg_reward}")
        
        # Get next action from learner and take action on robot
        action = learner.get_next_action()
        if action == 0:   # Decrease Position
            wait_time = arm.move_x_amount(MOTOR_ID, -MOVE_AMOUNT, min_pos=MOTOR_LIM_1, max_pos=MOTOR_LIM_2)
        elif action == 1: # Hold Position
            wait_time = 0
            pass
        elif action == 2: # Increase Position
            wait_time = arm.move_x_amount(MOTOR_ID, MOVE_AMOUNT, min_pos=MOTOR_LIM_1, max_pos=MOTOR_LIM_2)

        # Visualize
        plotter.update_data(pos, reward_next, learner.avg_reward, learner.softmax_probs, learner.last_action) 
        plotter.draw()

        # Wait f or motor to get to position and human to process visualization before next loop iteration
        time.sleep(wait_time + 0.5)
        print("\n")
    
    running = False