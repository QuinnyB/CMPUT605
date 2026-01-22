import time
from pynput import keyboard
from dynamixel_sdk import *

# --- Settings ---
DEVICE_NAME = 'COM15'  # Change to your actual COM port
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000

# Control Table Addresses
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112
ADDR_PRESENT_VELOCITY = 128

class Motor:
    def __init__(self, dxl_id, cw_limit, ccw_limit, speed, inverted=False):
        self.id = dxl_id
        self.cw_limit = cw_limit
        self.ccw_limit = ccw_limit
        self.speed = speed
        self.inverted = inverted
        self.moving = False

# --- Motor Definitions (Matching your Arduino Limits) ---
motors = {
    'shoulder':  Motor(1, 1500, 2450, 50),
    'elbow':     Motor(2, 1730, 2700, 50),
    'wristRot':  Motor(3, 1028, 3073, 60),
    'wristFlex': Motor(4, 848,  3248, 60, inverted=True),
    'hand':      Motor(5, 1650, 2658, 60)
}

# --- Keyboard Mapping ---
# Key : (Motor Name, Direction 1=CW, 2=CCW)
key_map = {
    'a': ('shoulder', 1), 'd': ('shoulder', 2),
    's': ('elbow', 1),    'w': ('elbow', 2),
    'j': ('wristRot', 1), 'l': ('wristRot', 2),
    'i': ('wristFlex', 1),'k': ('wristFlex', 2),
    'o': ('hand', 1),      'p': ('hand', 2)
}

# --- Setup SDK ---
portHandler = PortHandler(DEVICE_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

for m in motors.values():
    packetHandler.write1ByteTxRx(portHandler, m.id, ADDR_TORQUE_ENABLE, 1)

def stop_motor(m):
    # Get current position and stop there
    pos, _, _ = packetHandler.read4ByteTxRx(portHandler, m.id, ADDR_PRESENT_POSITION)
    vel, _, _ = packetHandler.read4ByteTxRx(portHandler, m.id, ADDR_PRESENT_VELOCITY)
    # Lead the stop slightly to prevent bounce
    stop_pos = pos + (vel // 10)
    packetHandler.write4ByteTxRx(portHandler, m.id, ADDR_PROFILE_VELOCITY, 60)
    packetHandler.write4ByteTxRx(portHandler, m.id, ADDR_GOAL_POSITION, stop_pos)
    m.moving = False

def on_press(key):
    try:
        k = key.char
        if k in key_map:
            name, direction = key_map[k]
            m = motors[name]
            if not m.moving:
                # Determine limit based on direction and inversion
                target = m.cw_limit if direction == 1 else m.ccw_limit
                if m.inverted: target = m.ccw_limit if direction == 1 else m.cw_limit
                
                packetHandler.write4ByteTxRx(portHandler, m.id, ADDR_PROFILE_VELOCITY, m.speed)
                packetHandler.write4ByteTxRx(portHandler, m.id, ADDR_GOAL_POSITION, target)
                m.moving = True
    except AttributeError: pass

def on_release(key):
    try:
        k = key.char
        if k in key_map:
            name, _ = key_map[k]
            stop_motor(motors[name])
    except AttributeError: pass
    if key == keyboard.Key.esc: return False # Exit on ESC

print("Control initialized. Use A/D, W/S, J/L, I/K, O/P. Press ESC to quit.")

# --- Start Listener ---
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()

# --- Cleanup ---
for m in motors.values():
    packetHandler.write1ByteTxRx(portHandler, m.id, ADDR_TORQUE_ENABLE, 0)
portHandler.closePort()