'''

'''

import time
import math
import random
from robotClass import MiniBento
from myoClass import MyoArmband, EMGProcessor
from visualizerClass import ACVisualizer
from keyboardHandlerClass import KeyPressHandler
from helperFunctions import *

# --- Configuration -----------------------------------------------------------------------------
# Robot arm:
COMM_PORT = 'COM13'     # Lab Mini Bento likes port 13, home likes 15
BAUDRATE = 1000000
MOTOR_VELO = 20
INITIAL_POSITIONS = {1: 2048, 2: 1800, 4: 2700, 5: 2780}
MOTOR_ID = 5            # Hand motor
HAND_POS_1 = 1750       # Hand closed position (encoder value)
HAND_POS_2 = 2650       # Hand open position (encoder value)

# Learning:


# Plotting:
visualizer_params = {
    "window_size": 100,
    "reward_range": [-1.0, 0.2],
    "action_mode": 'discrete',
    "action_key_map": {
        'a': 'CLOSE HAND',
        's': 'REST',
        'd': 'OPEN HAND'
    }
}

# --- Set up robot, learner, and visualizer  -------------------------------------------------------
with MiniBento(COMM_PORT, BAUDRATE, MOTOR_VELO, INITIAL_POSITIONS) as arm, MyoArmband() as myo:
    viz = ACVisualizer(visualizer_params)
    key_handler = KeyPressHandler(arm, MOTOR_ID)
    emg_processor = EMGProcessor()

    time.sleep(0.5)  # Delay to let arm get into initial position

    try:
        while viz.is_open():
            # Get the current key state ('a', 's', 'd', or None)
            active_key = key_handler.get_key()

            # Read and process data from myo
            myo_data = myo.get_data()
            emg_mav = emg_processor.process_frame(myo_data['emg'])
            emg_phasor_mag, emg_phasor_angle = get_phasor(emg_mav)
            print(f"EMG Phasor Magnitude: {emg_phasor_mag:.2f}, Angle: {emg_phasor_angle:.2f} degrees")

            # Create feature vector:
            feature_vector, flat_index, bin_indices = featurize_grid([emg_phasor_mag, emg_phasor_angle], [(0, 200), (-180, 180)], [3, 8])     
            print(f"Feature index: {flat_index}, Bin Indices: {bin_indices}")

            # 2. Run your Actor-Critic Logic
            # (Example dummy data)
            reward = 1.0 if active_key == 'a' else 0.0
            avg_reward = 0.5 
            probs = [0.1, 0.8, 0.1] # Policy output
            sampled_action = 1

            # 
            if active_key == 'a':
                arm.set_goal_pos(MOTOR_ID, HAND_POS_1)
            elif active_key == 's':
                arm.stop_motor(MOTOR_ID)
            elif active_key == 'd':
                arm.set_goal_pos(MOTOR_ID, HAND_POS_2)
        
            # 3. Update Visualizer
            # The visualizer will now show 'NO KEY PRESSED' if active_key is None
            viz.update_data(active_key, reward, avg_reward, probs, sampled_action)
            viz.draw()
        
            # Small sleep 
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Experiment stopped.")