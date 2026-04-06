'''
Main code for CMPUT 605 Final Project Interim Report 1
Exploration of ACRL for Myoelectric Control of a 3D Printed Robot Arm
Written by: Quinn Boser, with assistance from Google Gemini
April 2026
'''

import time
from robotClass import MiniBento
from myoClass import MyoArmband
from learnerClass import ACLearner_Discrete
from visualizerClass import ACVisualizer
from keyboardHandlerClass import KeyPressHandler
from helperFunctions import *

# --- Configuration -----------------------------------------------------------------------------
# Myo Armband:
MAV_WINDOW = 40        # Window size for moving average of EMG signals (in number of samples)

# Robot arm:
COMM_PORT = 'COM15'     # Lab Mini Bento likes port 13, home likes 15
BAUDRATE = 1000000
MOTOR_VELO = 20
INITIAL_POSITIONS = {1: 2048, 2: 1800, 4: 2700, 5: 2780}
MOTOR_ID = 5            # Hand motor
HAND_POS_1 = 1750       # Hand closed position (encoder value)
HAND_POS_2 = 2650       # Hand open position (encoder value)

# Learning:
EMG_MAG_RANGE = (0, 100)   # Expected range of EMG phasor magnitude (for featurization)
EMG_ANGLE_RANGE = (-180, 180)  # Expected range of EMG phasor angle (for featurization)
RANGES = [EMG_MAG_RANGE, EMG_ANGLE_RANGE]  # For featurization
NUM_MAG_BINS = 3        # Number of bins for EMG phasor magnitude
NUM_PHASOR_BINS = 8     # Number of bins for EMG phasor angle
BIN_COUNTS = [NUM_MAG_BINS, NUM_PHASOR_BINS]  # For featurization
agent_params = {
    "feature_vector_length": np.prod(BIN_COUNTS),
    "num_actions": 3,   # close [0], rest [1], or open [2]
    "avg_reward_alpha": 0.1,
    "critic_alpha": 0.9,
    "actor_alpha": 0.7
}
reward_next = 0.0   # Initialize reward for first loop (no action taken yet)

# Plotting:
visualizer_params = {
    "window_size": 100,
    "reward_range": [-1.2, 1.2],
    "show_feature_idx": True,
    "action_mode": 'discrete',
    "action_key_map": {
        'a': 'CLOSE HAND',
        's': 'REST',
        'd': 'OPEN HAND'
    }
}

# --- Set up robot, learner, and visualizer  -------------------------------------------------------
with MiniBento(COMM_PORT, BAUDRATE, MOTOR_VELO, INITIAL_POSITIONS) as arm, MyoArmband(MAV_WINDOW) as myo:
    key_handler = KeyPressHandler(arm, MOTOR_ID)
    learner = ACLearner_Discrete(agent_params)
    viz = ACVisualizer(visualizer_params)
    
    time.sleep(0.5)     # Delay to let arm get into initial position

    try:
        while viz.is_open():
            if key_handler.is_paused:
                viz.process_events() # Handle window events
                continue

            # Get active key ('a', 's', 'd', or None)
            active_key = key_handler.get_key()
            # Turn off learning mode if no key is active
            if active_key is None:
                learning = False
            else:
                learning = True

            # Read and process data from myo (observe state)
            emg_mav, _, _, _ = myo.get_data()
            emg_mag, emg_angle = get_phasor(emg_mav)
            print(f"EMG Phasor Magnitude: {emg_mag:.2f}, Angle: {emg_angle:.2f} degrees")

            # Create feature vector X:
            x_next, flat_index, bin_indices = featurize_grid([emg_mag, emg_angle], RANGES, BIN_COUNTS)     
            print(f"Feature index: {flat_index}, Bin Indices: {bin_indices}")

            # Update learner with reward from previous action (after first action)
            if learner.last_action is not None:
                match = check_action_match(action, active_key, viz.action_key_map)
                if learning:
                    reward_next = 1.0 if match else -1.0
                else:
                    reward_next = 0.0
                learner.update(reward_next, x_next, learning_enabled=learning)

            # Get next action from learner and take action on robot
            action = learner.get_next_action()
            if action == 0:
                arm.set_goal_pos(MOTOR_ID, HAND_POS_1)
            elif action == 1:
                arm.stop_motor(MOTOR_ID)
            elif action == 2:
                arm.set_goal_pos(MOTOR_ID, HAND_POS_2)
        
            # Update Visualizer
            viz.update_data(active_key, emg_mag, emg_angle, reward_next, 
                            learner.avg_reward, learner.softmax_probs, learner.last_action,
                            feature_index=flat_index)
            viz.draw()
        
            # Small sleep 
            time.sleep(0.2)

    except KeyboardInterrupt:
        print("Experiment stopped.")