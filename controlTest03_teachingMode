import time
from pynput import keyboard
from dynamixel_sdk import *

# --- Settings ---
DEVICE_NAME = 'COM13'  # Update to your port
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000

ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112

# Motor IDs (1-4 for arm, 5 for gripper)
MOTOR_IDS = [1, 2, 4, 5]
recorded_poses = []
torque_on = False

# --- SDK Setup ---
portHandler = PortHandler(DEVICE_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

def set_torque(state):
    global torque_on
    val = 1 if state else 0
    for m_id in MOTOR_IDS:
        packetHandler.write1ByteTxRx(portHandler, m_id, ADDR_TORQUE_ENABLE, val)
    torque_on = state
    print(f"--- Torque {'ENABLED' if state else 'DISABLED (Limp Mode)'} ---")

def record_pose():
    current_pose = []
    for m_id in MOTOR_IDS:
        pos, _, _ = packetHandler.read4ByteTxRx(portHandler, m_id, ADDR_PRESENT_POSITION)
        current_pose.append(pos)
    recorded_poses.append(current_pose)
    print(f"Pose #{len(recorded_poses)} recorded: {current_pose}")

def play_motion():
    if not recorded_poses:
        print("No poses recorded yet!")
        return
    
    print("Starting Playback...")
    set_torque(True)
    
    # Set a moderate speed for playback
    for m_id in MOTOR_IDS:
        packetHandler.write4ByteTxRx(portHandler, m_id, ADDR_PROFILE_VELOCITY, 20)

    for pose in recorded_poses:
        print(f"Moving to: {pose}")
        for i, m_id in enumerate(MOTOR_IDS):
            packetHandler.write4ByteTxRx(portHandler, m_id, ADDR_GOAL_POSITION, pose[i])
        
        # Wait a moment for the arm to reach the position
        time.sleep(1.5) 
    
    print("Playback Finished.")

# --- Keyboard Handling ---
def on_press(key):
    try:
        # 1. Handle "Special" Keys first (keys without a .char attribute)
        if key == keyboard.Key.esc:
            print("\nExiting Teaching Mode...")
            return False  # This is the "magic" line that stops the listener
            
        if key == keyboard.Key.space:
            record_pose()
            return # Keep the listener going

        # 2. Handle "Character" Keys (t, p, c)
        if hasattr(key, 'char'):
            if key.char == 't': # Toggle Torque
                set_torque(not torque_on)
            elif key.char == 'p': # Playback
                play_motion()
            elif key.char == 'c': # Clear
                recorded_poses.clear()
                print("All poses recorded have been cleared.")
                
    except Exception as e:
        print(f"Error in keyboard listener: {e}")

set_torque(False) # Start in limp mode
print("TEACHING MODE LOADED")
print("1. Move the arm by hand.")
print("2. Press SPACE to save a pose.")
print("3. Press 'T' to lock/unlock torque.")
print("4. Press 'P' to play back.")
print("5. Press 'C' to clear recording.")
print("Press ESC to quit.")

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

# Cleanup
set_torque(False)
portHandler.closePort()