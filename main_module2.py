from robotClass import MiniBento
from learnerClasses import TDLearner
from visualizerClass import RLVisualizer
from pynput import keyboard
import threading

# --- Set-up -----------------------------------------------------------------------------
# Robot arm:
COMM_PORT = 'COM13'  # Update this to your port!
BAUDRATE = 1000000
PROFILE_VELOCITY = 20
INITIAL_POSITIONS = {1: 2048, 2: 1980, 4: 2048, 5: 2780}


# Learning:
LOAD_THRESHOLD = 100  # Load threshold for cumulant
LOAD_THRESHOLD = 100  # Load threshold 
NUM_POS_BINS = 10   # For creating feature vector
NUM_VEL_BINS = 20    # For creating feature vector
GAMMA = 0.5         # Discount factor 
ALPHA = 1         # Learning rate

# Plotting:
plotter = RLVisualizer()

# Misc:
avg_update_time = 0
loop_count = 0
is_paused = False   # Global pause flag
running = True      # Control flag to stop threads

# Function pointers so the class can check our global flags
def get_paused(): return is_paused
def get_running(): return running

def on_press(key):
    global is_paused
    if key == keyboard.Key.space:
        is_paused = not is_paused
        print(f"*** {'PAUSED' if is_paused else 'RESUMED'} ***")

# -----------------------------------------------------------------------------------------
# --- Main Loop  --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
# The "Conductor" block
with MiniBento(COMM_PORT, BAUDRATE, PROFILE_VELOCITY, INITIAL_POSITIONS) as arm:
    brain = TDLearner(ALPHA, GAMMA, feature_vector_length=NUM_POS_BINS * NUM_VEL_BINS)
    plotter = RLVisualizer(window_size=200)

    # Start the Movement Thread targeting the class method
    mover = threading.Thread(
        target=arm.cycle_gripper, 
        args=(motorMovement, get_paused, get_running), 
        daemon=True
    )
    mover.start()

    while plotter.is_open():
        if is_paused:
            # Tell arm to stay put
            p, _, _ = arm.read_from_motor()
            arm.set_goal(motorMovement.HAND_ID, p)
            
            plt.pause(0.1) # Handle window events
            continue

        # 1. Sense
        pos, vel, load = arm.read_motor(motorMovement.HAND_ID)
        if pos is None: continue

        # 2. Learn (Math from module2_constantGamma)
        x = featurize(pos, vel, motorMovement, learningParams)
        c = cumulant_loadThreshold(load, learningParams.LOAD_THRESHOLD)
        
        brain.update(x, c)
        pred = brain.get_prediction(x)

        # 3. Visualize
        plotter.update_data(pos, vel, load, c, pred)
        plotter.draw()
    
    running = False