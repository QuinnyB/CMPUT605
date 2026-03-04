import time
from dynamixel_sdk import * # --- Settings ---
# On Windows, this is usually 'COM3', 'COM4', etc.
# On Linux/Mac, it's usually '/dev/ttyACM0' or '/dev/tty.usbmodem...'
DEVICE_NAME          = 'COM15' 
PROTOCOL_VERSION     = 2.0
BAUDRATE             = 1000000

# Control Table Addresses for XL330
ADDR_TORQUE_ENABLE   = 64
ADDR_GOAL_POSITION   = 116
ADDR_PRESENT_POSITION = 132
ADDR_PROFILE_VELOCITY = 112

HAND_ID              = 5
MOTOR_IDS            = [1, 2, 3, 4, 5]
INITIAL_POSITIONS    = {1: 2048, 2: 1980, 3: 1400, 4: 2048, 5: 2780}

# Initialize Handlers
portHandler = PortHandler(DEVICE_NAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open Port & Set Baudrate
if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
    print("Failed to connect. Check your COM port and Baudrate!")
    quit()

# --- Setup Motors ---
for m_id in MOTOR_IDS:
    # Enable Torque
    packetHandler.write1ByteTxRx(portHandler, m_id, ADDR_TORQUE_ENABLE, 1)
    # Set Speed (Profile Velocity)
    packetHandler.write4ByteTxRx(portHandler, m_id, ADDR_PROFILE_VELOCITY, 20)
    # Move to Initial Position
    packetHandler.write4ByteTxRx(portHandler, m_id, ADDR_GOAL_POSITION, INITIAL_POSITIONS[m_id])

print("Robot Ready.")

# --- Main Loop ---
try:
    while True:
        print("Closing Gripper...")
        packetHandler.write4ByteTxRx(portHandler, HAND_ID, ADDR_GOAL_POSITION, 1650)
        time.sleep(2)
        
        print("Opening Gripper...")
        packetHandler.write4ByteTxRx(portHandler, HAND_ID, ADDR_GOAL_POSITION, 2650)
        time.sleep(2)
# Press Ctrl + C to stop 
except KeyboardInterrupt:
    # Disable torque before closing so you can move the arm by hand
    for m_id in MOTOR_IDS:
        packetHandler.write1ByteTxRx(portHandler, m_id, ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort()
    print("Communication Closed.")