import math
import threading
from robotClass import MiniBento
from learnerClasses import TDLearner
from visualizerClass import RLVisualizer
from pynput import keyboard
from helperFunctions import featurize_pos_velo, cumulant_loadThreshold

# --- Set-up -----------------------------------------------------------------------------
# Robot arm:
COMM_PORT = 'COM15'  # Update this to your port!
BAUDRATE = 1000000
MOTOR_VELO = 20
INITIAL_POSITIONS = {1: 2048, 2: 1980, 4: 2048, 5: 2780}
HAND_ID = 5
HAND_POS_1= 1750
HAND_POS_2 = 2650
POS_RANGE = HAND_POS_2 - HAND_POS_1
WAIT_TIME = math.ceil(((POS_RANGE) / (4096 * 0.229 * MOTOR_VELO)) * 60)

# Learning:
LOAD_THRESHOLD = 100  # Load threshold for cumulant
NUM_POS_BINS = 10   # For creating feature vector
NUM_VEL_BINS = 20   # For creating feature vector
GAMMA = 0.5         # Discount factor 
ALPHA = 0.7         # Learning rate
VERIFIER_BUFFER_LENGTH = math.ceil(5*(1/(1-GAMMA)))  # Number of steps to look back at for verifier

# Misc:
avg_update_time = 0
loop_count = 0
is_paused = False   # Global pause flag
running = True      # Control flag to stop threads

# Function pointers so the robot class can check global flags
def get_paused(): return is_paused
def get_running(): return running

# -----------------------------------------------------------------------------------------
# --- Main Loop  --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
with MiniBento(COMM_PORT, BAUDRATE, MOTOR_VELO, INITIAL_POSITIONS) as arm:
    learner = TDLearner(ALPHA, GAMMA, feature_vector_length=NUM_POS_BINS * NUM_VEL_BINS)
    plotter = RLVisualizer(window_size=200)

    # Define on_press inside the with block to access learner
    def on_press(key):
        global is_paused
        if key == keyboard.Key.space:
            is_paused = not is_paused
            print(f"*** {'PAUSED' if is_paused else 'RESUMED'} ***")
            if is_paused:
                # Stop motor where it is
                p, _, _ = arm.read_from_motor(HAND_ID)
                arm.set_goal_pos(HAND_ID, p)
        else:
            try:
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
        x_next = featurize_pos_velo(pos, vel, HAND_POS_1, HAND_POS_2, MOTOR_VELO, NUM_POS_BINS, NUM_VEL_BINS)
        c_next = cumulant_loadThreshold(load, LOAD_THRESHOLD)
        
        # Update TD learner and get prediction
        learner.update(x_next, c_next)
        pred = learner.get_prediction(x_next)

        # 3. Visualize
        plotter.update_data(pos, vel, load, c_next, pred)
        if loop_count > VERIFIER_BUFFER_LENGTH:
            plotter.update_verifier(VERIFIER_BUFFER_LENGTH, GAMMA)
        plotter.draw()
    
    running = False