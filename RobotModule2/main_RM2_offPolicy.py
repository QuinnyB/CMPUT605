''' 
Main code for CMPUT 605 Robot Module 2 - Objectibve 2: Off-policy TD Learning
Written by: Quinn Boser, with assistance from Google Gemini 
Feb. 2026
'''

import math
import time
import threading
from robotClass import MiniBento
from learnerClass import TDLearner
from visualizerClass import TDVisualizer
from pynput import keyboard
from helperFunctions import *

# --- Configuration -----------------------------------------------------------------------------
# Robot arm:
COMM_PORT = 'COM15'     # Lab Mini Bento likes port 13, home likes 15
BAUDRATE = 1000000
MOTOR_VELO = 20
INITIAL_POSITIONS = {1: 2048, 2: 1700, 4: 2600, 5: 2780}
MOTOR_ID = 4
MOTOR_LIM_1= 2150
MOTOR_LIM_2 = 3200
MOTOR_RANGE = MOTOR_LIM_2 - MOTOR_LIM_1
WAIT_TIME = math.ceil(((MOTOR_RANGE) / (4096 * 0.229 * MOTOR_VELO)) * 60)

# Learning:
LOAD_THRESHOLD = -15  # Load threshold for bump detection (applied directionally)
NUM_POS_BINS = 10   # For creating feature vector
NUM_VEL_BINS = 3    # For creating feature vector
GAMMA = 1.0         # Default termination signal
ALPHA = 0.4         # Learning rate
LAMBDA_ = 0.0       # Eligibility trace decay rate
VERIFIER_BUFFER_LENGTH = 20
INITIAL_W = 10.0

# Plotting
pred_plot_scale = 1

# Misc:
loop_count = 0
is_paused = False   # Global pause flag
running = True      # Control flag to stop threads
avg_loop_time = 0   # Running average of loop iteration time

# Function pointers so the robot class can check global flags
def get_paused(): return is_paused
def get_running(): return running

# --- Set up robot, learner, and visualizer  -------------------------------------------------------
with MiniBento(COMM_PORT, BAUDRATE, MOTOR_VELO, INITIAL_POSITIONS) as arm:
    learner = TDLearner(ALPHA, GAMMA, feature_vector_length=NUM_POS_BINS*NUM_VEL_BINS, 
                        history_length=VERIFIER_BUFFER_LENGTH, initial_w=INITIAL_W, lambda_=LAMBDA_)
    plotter = TDVisualizer(window_size=200, plotFeatureIndex=True, c_label='Gamma')

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
        else:
            try:
                # 'a' key to toggle learning on/off
                if key.char == 'a':
                    if learner.alpha == 0:
                        learner.alpha = ALPHA
                        print("*** LEARNER ALPHA RESTORED ***")
                    else:
                        learner.alpha = 0
                        print("*** LEARNER ALPHA SET TO 0 ***")
            except AttributeError:
                pass

    # Start the Movement Thread
    mover = threading.Thread(
        target = arm.random_walk,
        args=(MOTOR_ID, MOTOR_LIM_1, MOTOR_LIM_2, get_paused, get_running), 
        daemon=True
    )
    mover.start()

    # Start the Keyboard Listener Thread
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

# -----------------------------------------------------------------------------------------
# --- Main Loop  --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
    while plotter.is_open():
        loop_start_time = time.perf_counter()  # Track loop iteration time
        
        if is_paused:
            plotter.process_events() # Handle window events
            continue
        
        # Increment loop count
        loop_count += 1

        # Get next state from the robot
        pos, vel, load = arm.read_from_motor(MOTOR_ID)
        if pos is None: continue

        # Convert next state into feature vector, cumulant, and gamma
        x_next, feature_idx, _, _ = featurize_pos_velo(pos, vel, MOTOR_LIM_1, MOTOR_LIM_2, MOTOR_VELO, NUM_POS_BINS, NUM_VEL_BINS)
        c_next, gamma_next = get_c_and_gamma_loadCountdown(load, LOAD_THRESHOLD, GAMMA)

        # Get imprortance sampling ratio (1 if moving down toward bump (negative velocity), 0 if moving up away from bump)
        if vel < 0:
            importance_ratio = 1.0
        # Clear eligibility traces if not in line with target policy
        else:
            importance_ratio = 0.0
            learner.clear_eligibility_traces()     

        # Update TD learner and get next prediction
        pred = learner.update_withEligibilityTraces(x_next, c_next, gamma_next, importance_ratio)

        # Visualize
        plotter.update_data(pos, vel, load, gamma_next, pred*pred_plot_scale, feature_idx)
        if loop_count > VERIFIER_BUFFER_LENGTH:
            expected_pred, idx_back = learner.get_verifier_data()
            plotter.update_verifier(expected_pred*pred_plot_scale, idx_back)
        plotter.draw()

        # Update running average of loop time
        loop_elapsed = time.perf_counter() - loop_start_time
        avg_loop_time = avg_loop_time + (loop_elapsed-avg_loop_time) / (loop_count)
        # if loop_count % 100 == 0:
        #     print(f"Avg Loop Time: {avg_loop_time:.4f} sec")

        time.sleep(0.01)  # Small delay to control update rate
  
    running = False