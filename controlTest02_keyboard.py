import time
import threading
import matplotlib.pyplot as plt
from collections import deque
from pynput import keyboard
from dynamixel_sdk import *

# --- Settings ---
DEVICE_NAME = 'COM13'  # Change to your actual COM port
PROTOCOL_VERSION = 2.0
BAUDRATE = 1000000
WINDOW_SIZE = 100  # Number of data points to show on the screen at once

# Control Table Addresses
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112
ADDR_PRESENT_VELOCITY = 128

class Motor:
    def __init__(self, dxl_id, cw_limit, ccw_limit, speed, name, color, inverted=False):
        self.id = dxl_id
        self.cw_limit = cw_limit
        self.ccw_limit = ccw_limit
        self.speed = speed
        self.name = name
        self.color = color
        self.inverted = inverted
        self.moving = False
        # Deque stores the history of positions for the line plot
        self.history = deque([2048] * WINDOW_SIZE, maxlen=WINDOW_SIZE)

# --- Motor Definitions (Matching your Arduino Limits) ---
motors = {
    'shoulder':  Motor(1, 1500, 2450, 10, 'Shoulder', 'red'),
    'elbow':     Motor(2, 1730, 2700, 10, 'Elbow', 'blue'),
    'wristFlex': Motor(4, 848,  3248, 10, 'Wrist Flex', 'orange', inverted=True),
    'hand':      Motor(5, 1650, 2658, 10, 'Hand', 'purple', inverted=True)
}

# --- Keyboard Mapping ---
# Key : (Motor Name, Direction 1=CW, 2=CCW)
key_map = {
    'a': ('shoulder', 1), 'd': ('shoulder', 2),
    's': ('elbow', 1),    'w': ('elbow', 2),
    'i': ('wristFlex', 1),'k': ('wristFlex', 2),
    'j': ('hand', 1),      'l': ('hand', 2)
}

# --- Setup SDK ---
portHandler = PortHandler(DEVICE_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)
portHandler.openPort()
portHandler.setBaudRate(BAUDRATE)

for m in motors.values():
    packetHandler.write1ByteTxRx(portHandler, m.id, ADDR_TORQUE_ENABLE, 1)

# --- Logic Functions ---
def stop_motor(m):
    pos, _, _ = packetHandler.read4ByteTxRx(portHandler, m.id, ADDR_PRESENT_POSITION)
    packetHandler.write4ByteTxRx(portHandler, m.id, ADDR_GOAL_POSITION, pos)
    m.moving = False

def on_press(key):
    try:
        if key.char in key_map:
            name, direction = key_map[key.char]
            m = motors[name]
            if not m.moving:
                target = m.cw_limit if direction == 1 else m.ccw_limit
                if m.inverted: target = m.ccw_limit if direction == 1 else m.cw_limit
                packetHandler.write4ByteTxRx(portHandler, m.id, ADDR_PROFILE_VELOCITY, m.speed)
                packetHandler.write4ByteTxRx(portHandler, m.id, ADDR_GOAL_POSITION, target)
                m.moving = True
    except: pass

def on_release(key):
    try:
        if key.char in key_map:
            m = motors[key_map[key.char][0]]
            pos, _, _ = packetHandler.read4ByteTxRx(portHandler, m.id, ADDR_PRESENT_POSITION)
            packetHandler.write4ByteTxRx(portHandler, m.id, ADDR_GOAL_POSITION, pos)
            m.moving = False
    except: pass

# --- Start Keyboard Thread ---
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# --- Live Plotting Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(12, 6))
lines = {}

# Create a line for each motor
for name, m in motors.items():
    line, = ax.plot(list(m.history), label=m.name, color=m.color, linewidth=2)
    lines[name] = line

ax.set_ylim(0, 4095)
ax.set_xlim(0, WINDOW_SIZE - 1)
ax.set_ylabel('Position (0-4095)')
ax.set_xlabel('Time (Samples)')
ax.set_title('Live Motor Position Scope')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

print("Controls: A/D, W/S, J/L, I/K, O/P. Close plot to exit.")

# --- Main Loop ---
try:
    while plt.fignum_exists(fig.number):
        for name, m in motors.items():
            # Read hardware
            pos, _, _ = packetHandler.read4ByteTxRx(portHandler, m.id, ADDR_PRESENT_POSITION)
            if pos is not None:
                m.history.append(pos)
            
            # Update the specific line on the plot
            lines[name].set_ydata(list(m.history))
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01) # High speed update

except KeyboardInterrupt:
    pass

# --- Cleanup ---
print("Shutting down...")
listener.stop()
for m in motors.values():
    packetHandler.write1ByteTxRx(portHandler, m.id, ADDR_TORQUE_ENABLE, 0)
portHandler.closePort()