'''
Main code for CMPUT 605 Final Project
Exploration of ACRL for Myoelectric Control of a 3D Printed Robot Arm
Bonus Exploration - 2 DoF myoelectric control 
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
# General:
IMU_CTRL_ON = False         # Whether to use IMU for controlling shoulder and elbow (True) or just use EMG for hand control (False)

# Myo Armband:
EMG_MAV_WINDOW = 40        # Window size for moving average of EMG signals (in number of samples)r     
IMU_MAV_WINDOW = 10        # Window size for moving average of IMU signals (in number of samples)
# Calibration steps: Instruction, dictionary_key, angle_index, joint_label
# For each joint, first step should be expected to correspond to smaller encoder value for motor
calibration_steps = [
    ("Internally rotate shoulder to comfortable limit", "int_rot", 0, "shoulder"),
    ("Externally rotate shoulder to comfortable limit", "ext_rot", 0, "shoulder"),
    ("Extend your elbow to comfortable limit",          "ext",     2, "elbow"),
    ("Flex your elbow to comfortable limit",            "flex",    2, "elbow"),
]

# Robot arm:
COMM_PORT = 'COM15'     # Lab Mini Bento likes port 13, home likes 15
BAUDRATE = 1000000
MOTOR_VELO = 20
INITIAL_POSITIONS = {1: 2048, 2: 1800, 4: 2700, 5: 2780}
SHO_ID = 1                  # Shoulder motor ID
SHO_LIMS = [1500, 2500]     # Shoulder motor limits (encoder values [CCW, CW])
ELB_ID = 2                  # Elbow motor ID    
ELB_LIMS = [1300, 2700]     # Elbow motor limits (encoder values [Ext, Flex])
WRIST_ID = 4                # Wrist motor ID
WRIST_LIMS = [2700, 3000]   # Wrist motor limits 
HAND_ID = 5                 # Hand motor ID
HAND_LIMS = [1750, 2650]    # Hand motor limits (encoder values [Open Close])

# Learning:
EMG_MAG_RANGE = [0, 100]   # Expected range of EMG phasor magnitude (for featurization)
EMG_ANGLE_RANGE = [-180, 180]  # Expected range of EMG phasor angle (for featurization)
NUM_MAG_BINS = 3        # Number of bins for EMG phasor magnitude
NUM_ANGLE_BINS = 8      # Number of bins for EMG phasor angle
NUM_SHO_BINS = 5        # Number of bins for shoulder position
agent_params = {
    "num_actions": 5,   # close [0], rest [1], or open [2], wrist down [3], wrist up [4]
    "avg_reward_alpha": 0.1,
    "critic_alpha": 0.9,
    "actor_alpha": 0.7,
    "unlearn_rate": 0.05, 
    "tile_coder_config": {
        'num_tilings': 16,
        'ranges': [EMG_MAG_RANGE, EMG_ANGLE_RANGE, SHO_LIMS],
        'bins_per_dim': [NUM_MAG_BINS, NUM_ANGLE_BINS, NUM_SHO_BINS],  
        'wrap_dims': [False, True, False],
        'use_hashing': False
    }
}
reward_next = 0.0   # Initialize reward for first loop (no action taken yet)

# Plotting:
visualizer_params = {
    "window_size": 100,
    "reward_range": [-1.2, 1.2],
    "show_feature_idx": True,
    "action_mode": 'discrete',
    "action_key_map": {
        's': 'REST',
        'a': 'CLOSE HAND',
        'd': 'OPEN HAND',
        'w': 'Wrist Down',
        'x': 'Wrist Up'
    }
}

# --- Set up robot, myo armband, learner, and visualizer  -------------------------------------------------------
with MiniBento(COMM_PORT, BAUDRATE, MOTOR_VELO, INITIAL_POSITIONS) as arm, MyoArmband(EMG_MAV_WINDOW, IMU_MAV_WINDOW) as myo:
    # Set up keyboard handler
    key_handler = KeyPressHandler(arm, HAND_ID)

    # Run Calibration process for IMU control of shoulder and elbow
    if IMU_CTRL_ON:
        imu_cal_angles, joint_mults = myo.run_interactive_calibration(key_handler, calibration_steps)

    # Setup ACRL learner and visualizer
    learner = ACLearner_Discrete(agent_params)
    viz = ACVisualizer(visualizer_params)

# --- Main Loop: Read Myo data, take action on robot, update learner, and update visualizer  ----------------------
    try:
        while viz.is_open():
            # Check for re-calibration request
            if IMU_CTRL_ON and hasattr(key_handler, 'recalibrate_requested') and key_handler.recalibrate_requested:
                # Return robot to initial positions before re-calibration
                arm.return_to_initial_positions()
                # Re-run calibration process
                imu_cal_angles, joint_mults = myo.run_interactive_calibration(key_handler, calibration_steps)
                # Reset the flag so we don't loop calibration forever
                key_handler.recalibrate_requested = False
                continue # Skip the rest of this loop iteration to start fresh


            if key_handler.is_paused:
                viz.process_events() # Handle window events
                continue

            # Read emg and angle data from myo (has been averaged over moving window)
            emg_mav, angles, _, _, _ = myo.get_data()

            # Set Mini Bento shoulder and elbow position based on IMU angles
            if IMU_CTRL_ON:
                target_s = np.interp(joint_mults["shoulder"] * angles[0], 
                            [imu_cal_angles["int_rot"], imu_cal_angles["ext_rot"]], 
                            SHO_LIMS)
                target_e = np.interp(joint_mults["elbow"] * angles[2], 
                            [imu_cal_angles["ext"], imu_cal_angles["flex"]], 
                            ELB_LIMS)
                print(f"Target SHO: {target_s:.0f}, Target ELB: {target_e:.0f}")
                arm.set_goal_pos(SHO_ID, int(target_s))
                arm.set_goal_pos(ELB_ID, int(target_e))

            # Get active keyboard key ('a', 's', 'd', or None)
            active_key = key_handler.get_key()
            # Turn off learning mode if no key is active
            if active_key is None:
                learning = False
            else:
                learning = True

            # Compute phasor representation of EMG MAV
            emg_mag, emg_angle = get_phasor(emg_mav)
            # print(f"EMG Phasor Magnitude: {emg_mag:.2f}, Angle: {emg_angle:.2f} degrees")

            # Update learner with reward from previous action (after first action)
            if learner.last_action is not None:
                match = check_action_match(action, active_key, viz.action_key_map)
                if learning:
                    reward_next = 1.0 if match else -1.0
                else:
                    reward_next = 0.0
                # state = [emg_mag, emg_angle]
                state = [emg_mag, emg_angle, target_s if IMU_CTRL_ON else 2000]
                learner.update(reward_next, state, learning_enabled=learning)

            # Get next action from learner and take action on robot
            action = learner.get_next_action()
            if action == 0:
                arm.stop_motor(WRIST_ID)
                arm.stop_motor(HAND_ID)
            elif action == 1:
                arm.set_goal_pos(HAND_ID, HAND_LIMS[0])
            elif action == 2:
                arm.set_goal_pos(HAND_ID, HAND_LIMS[1])
            elif action == 3: 
                arm.set_goal_pos(WRIST_ID, WRIST_LIMS[0])
            elif action == 4:
                arm.set_goal_pos(WRIST_ID, WRIST_LIMS[1])

        
            # Update Visualizer
            if learner.cur_active_indices:
                base_indices = [idx % learner.tc.tiles_per_tiling for idx in learner.cur_active_indices]
                refined_id = np.mean(base_indices)
            else:
                refined_id = None
            viz.update_data(active_key, emg_mag, emg_angle, reward_next, 
                            learner.avg_reward, learner.softmax_probs, learner.last_action,
                            feature_index=refined_id)
            viz.draw()
        
            # Small sleep 
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Experiment stopped.")