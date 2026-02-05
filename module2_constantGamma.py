import time
import threading
import math
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard
from collections import deque
from dynamixel_sdk import *
from dataclasses import dataclass
from robotModuleFunctions import *

# --- Configuration -----------------------------------------------------------------------
# Motor communication:
COMM_PORT = 'COM13'  # Update this to your port!
BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0
MOTOR_IDS = [1, 2, 4, 5]
INITIAL_POSITIONS = {1: 2048, 2: 1980, 4: 2048, 5: 2780}

# Register Addresses
@dataclass
class MotorAddresses:    
    TORQUE_ENABLE: int = 64
    GOAL_POSITION: int = 116
    PROFILE_VELOCITY: int = 112
    PRESENT_LOAD: int = 126
    PRESENT_VELOCITY: int = 128
    PRESENT_POSITION: int = 132

# Motor movement:  
@dataclass
class MotorMovement:
    MOTOR_VELO: int = 20
    HAND_ID: int = 5
    HAND_POS_1: int = 1750
    HAND_POS_2: int = 2650
    POS_RANGE = HAND_POS_2 - HAND_POS_1
    WAIT_TIME = math.ceil(((POS_RANGE) / (4096 * 0.229 * MOTOR_VELO)) * 60)

# Learning parameters:
@dataclass
class LearningParams:
    MAX_LOAD: int = 300      # Max expected absolute load for normalization
    LOAD_THRESHOLD: int = 100  # Load threshold 
    NUM_POS_BINS: int = 10   # For creating feature vector
    NUM_VEL_BINS: int = 20    # For creating feature vector
    GAMMA: float = 0.5         # Discount factor 
    ALPHA: float = 1         # Learning rate

motorAddresses = MotorAddresses()
motorMovement = MotorMovement()
learningParams = LearningParams()

verifier_buffer_length = math.ceil(5*(1/(1-learningParams.GAMMA)))  # Number of steps to look back at for verification
verifier_indices = np.arange(verifier_buffer_length)    # Create an array of indices [0, 1, 2, ..., verifier_buffer_length-1]
w = np.zeros(learningParams.NUM_POS_BINS * learningParams.NUM_VEL_BINS)  # Weight vector initialization
x_prev = np.zeros(learningParams.NUM_POS_BINS * learningParams.NUM_VEL_BINS, dtype=int)  # Previous feature vector

# Plotting:
WINDOW_SIZE = 200 # How many points to show on the screen
pos_history = deque([2048] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
vel_history = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
load_history = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
cumulant_history = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
pred_history = deque([0] * WINDOW_SIZE, maxlen=WINDOW_SIZE)
verifier_history = deque([np.nan] * WINDOW_SIZE, maxlen=WINDOW_SIZE)

# Misc:
avg_update_time = 0
loop_count = 0
is_paused = False  # Global pause flag
running = True # Control flag to stop threads

# Background thread to move the gripper:
def move_logic():
    # This function runs in the background to cycle the hand.
    global running
    while running:
        if is_paused:
            time.sleep(0.1)
            continue
        # Move to Close
        packetHandler.write4ByteTxRx(portHandler, motorMovement.HAND_ID, motorAddresses.GOAL_POSITION, motorMovement.HAND_POS_1)
        time.sleep(motorMovement.WAIT_TIME)
        # Move to Open
        packetHandler.write4ByteTxRx(portHandler, motorMovement.HAND_ID, motorAddresses.GOAL_POSITION, motorMovement.HAND_POS_2)
        time.sleep(motorMovement.WAIT_TIME)

# Kayboard listener for pausing/resuming
def on_press(key):
    global is_paused
    if key == keyboard.Key.space:
        is_paused = not is_paused
        status = "PAUSED" if is_paused else "RESUMED"
        print(f"\n*** {status} ***")
        if is_paused:
            # Tell motor to stop exactly where it is immediately
            pos, _, _ = packetHandler.read4ByteTxRx(portHandler, motorMovement.HAND_ID, motorAddresses.PRESENT_POSITION)
            packetHandler.write4ByteTxRx(portHandler, motorMovement.HAND_ID, motorAddresses.GOAL_POSITION, pos)

# --- Communitcation and Motor Setup ------------------------------------------------------
portHandler = PortHandler(COMM_PORT)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
    print("Failed to open port. Check connection!")
    quit()

for m_id in MOTOR_IDS:
    # Enable Torque
    packetHandler.write1ByteTxRx(portHandler, m_id, motorAddresses.TORQUE_ENABLE, 1)
    # Set Speed (Profile Velocity)
    packetHandler.write4ByteTxRx(portHandler, m_id, motorAddresses.PROFILE_VELOCITY, motorMovement.MOTOR_VELO)
    # Move to Initial Position
    packetHandler.write4ByteTxRx(portHandler, m_id, motorAddresses.GOAL_POSITION, INITIAL_POSITIONS[m_id])
print("Robot Ready.")


# Start the threads
listener = keyboard.Listener(on_press=on_press)
listener.start()
mover = threading.Thread(target=move_logic, daemon=True)
mover.start()

# --- Live Plotting Setup -----------------------------------------------------------------
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(10, 6))

# Axis 1: Present Position (top plot, left axis)
axs[0].set_xlabel('Time (Samples)')
axs[0].set_ylabel('Motor Position (0-4095)', color='blue')
line_pos, = axs[0].plot(list(pos_history), color='blue', label='Position')
axs[0].tick_params(axis='y', labelcolor='blue')
axs[0].set_ylim(0, 4095)

# Axis 2: Present Velocity (top plot, right axis)
axs0_left = axs[0].twinx() # This creates the second Y-axis
axs0_left.set_ylabel('Velocity', color='green')
line_vel, = axs0_left.plot(list(vel_history), color='green', alpha=0.6, label='Velocity')
axs0_left.tick_params(axis='y', labelcolor='green')
axs0_left.set_ylim(-100, 100)

# Axis 3: Present Load (middle plot)
axs[1].set_ylabel('Raw Load (0-2000)', color='red')
line_load, = axs[1].plot(list(load_history), color='red', alpha=0.6, label='Load')
axs[1].tick_params(axis='y', labelcolor='red')
axs[1].set_ylim(-400, 400) 

# Axis 4: Normalized load and prediction (Bottom plot)
axs[2].set_ylabel('Cumulant Signal / Prediction', color='purple')
line_c, = axs[2].plot(list(cumulant_history), color='orange', alpha=0.6, label='Cumulant Signal')
line_pred, = axs[2].plot(list(pred_history), color='purple', alpha=0.6, label='Prediction')
line_verifier, = axs[2].plot(list(verifier_history), color='brown', alpha=0.6, label='Verifier')
axs[2].tick_params(axis='y', labelcolor='purple')
axs[2].set_ylim(-0.5, 1.5)

fig.tight_layout()
axs[0].grid(True, alpha=0.3)
plt.title(f"Motor {motorMovement.HAND_ID}: Position, Velocity, and Load")

print("Starting live plot. Close the window to stop.")

# -----------------------------------------------------------------------------------------
# --- Main Loop  --------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
try:
    while plt.fignum_exists(fig.number):
        start_time = time.perf_counter()
        if is_paused:
            # Still flush events - allows the window to process the "Close" click
            fig.canvas.flush_events()
            time.sleep(0.1)
            continue

        # Read actual position, velocity, and load from the motor
        pos, vel, load = read_from_motor(packetHandler, portHandler, motorMovement.HAND_ID, motorAddresses)
        if pos is not None:
            pos_history.append(pos)  
            vel_history.append(vel)  
            load_history.append(load)

        # Covert pos and vel into feature vector
        x = featurize(pos, vel, motorMovement, learningParams)

        # Convert load into signal of interest (cumulant)
        c = cumulant_loadThreshold(load, learningParams.LOAD_THRESHOLD)
        cumulant_history.append(c)

        # Calculate delta
        delta = c + learningParams.GAMMA*w@x - w@x_prev
 
        # Update W
        w = w + learningParams.ALPHA*delta*x_prev

        # Store state
        x_prev = x

        # Calculate prediction
        pred = w@x * (1-learningParams.GAMMA)
        pred_history.append(pred) # This might be updating the wrong index, need to think about it more

        # Update plot lines
        line_pos.set_ydata(list(pos_history))
        line_vel.set_ydata(list(vel_history))
        line_load.set_ydata(list(load_history))
        line_c.set_ydata(list(cumulant_history))
        line_pred.set_ydata(list(pred_history))
        
        # Refresh the plot
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        time.sleep(0.01) # Small delay to control update rate
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        avg_update_time = avg_update_time + (elapsed_time - avg_update_time) / (loop_count + 1)
        loop_count += 1
        # if loop_count % 50 == 0:
        #     print(f"Avg Loop Time: {avg_update_time:.4f} sec")
        #     print(w)

        if loop_count > verifier_buffer_length:
            # First push a nan onto the end of verifier history
            verifier_history.append(np.nan)
            # Compute expected prediction for verifier_buffer_length steps in the past:
            verifier_buffer = list(cumulant_history)[-verifier_buffer_length:]
            expected_pred = np.sum(verifier_buffer * (learningParams.GAMMA ** verifier_indices)) * (1-learningParams.GAMMA)
            verifier_history[-(verifier_buffer_length+1)] = expected_pred
            line_verifier.set_ydata(list(verifier_history))


except KeyboardInterrupt:
    pass

# --- Cleanup ---
print("\nShutting down...")
running = False
# Disable torque before closing so you can move the arm by hand
for m_id in MOTOR_IDS:
    packetHandler.write1ByteTxRx(portHandler, m_id, motorAddresses.TORQUE_ENABLE, 0)
portHandler.closePort()
print("Communication Closed.")
portHandler.closePort()
plt.close()