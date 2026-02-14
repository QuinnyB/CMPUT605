''' 
Main code for CMPUT 605 Robot Module 2 - Objectibve 1: On-Policy TD Learning with State-conditional Gamma
Written by: Quinn Boser, with assistance from Google Gemini 
Feb. 2026
'''

import math
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
HAND_ID = 5
HAND_POS_1= 1750
HAND_POS_2 = 2650
POS_RANGE = HAND_POS_2 - HAND_POS_1
WAIT_TIME = math.ceil(((POS_RANGE) / (4096 * 0.229 * MOTOR_VELO)) * 60)

# Learning:
LOAD_THRESHOLD = 100  # Load threshold for cumulant (applied as absolute value)
NUM_POS_BINS = 10   # For creating feature vector
NUM_VEL_BINS = 20   # For creating feature vector
GAMMA = 0.7         # Default (higher) termination signal
GAMMA_LOW = 0.01    # Termination signal while load exceeds threshold
ALPHA = 0.4         # Learning rate
LAMBDA_ = 0.8       # Eligibility trace decay rate
VERIFIER_BUFFER_LENGTH = math.ceil(5*(1/(1-GAMMA)))  # Number of steps to look back at for verifier

# Plotting
pred_plot_scale = (1-(GAMMA+GAMMA_LOW)/2)

# Misc:
avg_update_time = 0
loop_count = 0
is_paused = False   # Global pause flag
running = True      # Control flag to stop threads

# Function pointers so the robot class can check global flags
def get_paused(): return is_paused
def get_running(): return running

# --- Set up robot, learner, and visualizer  -------------------------------------------------------
with MiniBento(COMM_PORT, BAUDRATE, MOTOR_VELO, INITIAL_POSITIONS) as arm:
    # learner = TDLearner(ALPHA, GAMMA, feature_vector_length=NUM_POS_BINS*NUM_VEL_BINS, history_length=VERIFIER_BUFFER_LENGTH)
    learner = TDLearner(ALPHA, GAMMA, feature_vector_length=NUM_POS_BINS*NUM_VEL_BINS, 
                        history_length=VERIFIER_BUFFER_LENGTH, lambda_=LAMBDA_)
    plotter = TDVisualizer(window_size=200)

    # Define keyboard event handler
    def on_press(key):
        global is_paused
        # Spacebar to Pause/Resume
        if key == keyboard.Key.space:
            is_paused = not is_paused
            print(f"*** {'PAUSED' if is_paused else 'RESUMED'} ***")
            if is_paused:
                # Stop motor where it is
                p, _, _ = arm.read_from_motor(HAND_ID)
                arm.set_goal_pos(HAND_ID, p)
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
        target = arm.cycle_motor, 
        args=(HAND_ID, HAND_POS_1, HAND_POS_2, WAIT_TIME, get_paused, get_running), 
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
        if is_paused:
            plotter.process_events() # Handle window events
            continue
        
        # Increment loop count
        loop_count += 1

        # Get next state from the robot
        pos, vel, load = arm.read_from_motor(HAND_ID)
        if pos is None: continue

        # Convert next state into feature vector and cumulant
        x_next, feature_idx, pos_bin, vel_bin = featurize_pos_velo(pos, vel, HAND_POS_1, HAND_POS_2, MOTOR_VELO, NUM_POS_BINS, NUM_VEL_BINS)
        c_next = get_cumulant_absLoadThreshold(load, LOAD_THRESHOLD)
        print(f"Step {loop_count}: pos={pos}, pos_bin={pos_bin}, vel={vel}, vel_bin={vel_bin}, load={load}, c={c_next}, feature_idx={feature_idx}")

        # State conditional gamma:
        gamma_next = GAMMA_LOW if c_next==1 else GAMMA

        # Update TD learner and get next prediction
        pred = learner.update_withEligibilityTraces(x_next, c_next, gamma_next=gamma_next)

        # Visualize
        plotter.update_data(pos, vel, load, c_next, pred*pred_plot_scale)  # Scale prediction for plotting
        if loop_count > VERIFIER_BUFFER_LENGTH:
            expected_pred, idx_back = learner.get_verifier_data()
            plotter.update_verifier(expected_pred*pred_plot_scale, idx_back)    # Scale verifier for plotting
        plotter.draw()

        time.sleep(0.1)  # Small delay 
    
    running = False