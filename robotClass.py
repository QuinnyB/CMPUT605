import threading
import time
from dynamixel_sdk import *
from helperFunctions import to_signed_32, to_signed_16 

class MiniBento:
    def __init__(self, port_name, baudrate, motor_velo, iniital_positions):
        self.portHandler = PortHandler(port_name)
        self.baudrate = baudrate
        self.motor_velo = motor_velo
        self.initial_positions = iniital_positions
        self.packetHandler = PacketHandler(2.0) # Using protocol version 2.0
        self.addr_torque_enable = 64
        self.addr_goal_position = 116
        self.addr_profile_velocity = 112
        self.addr_present_load = 126
        self.addr_present_velocity = 128
        self.addr_present_position = 132
        self.motor_ids = [1, 2, 4, 5]
        self.lock = threading.Lock() 
        
    def __enter__(self):
        if not self.portHandler.openPort() or not self.portHandler.setBaudRate(self.baudrate):
            raise ConnectionError("Could not connect to Dynamixels.")
        
        for m_id in self.motor_ids:
            # Enable Torque
            self.packetHandler.write1ByteTxRx(self.portHandler, m_id, self.addr_torque_enable, 1)
            # Set Speed (Profile Velocity)
            self.packetHandler.write4ByteTxRx(self.portHandler, m_id, self.addr_profile_velocity, self.motor_velo)
            # Move to Initial Position
            self.packetHandler.write4ByteTxRx(self.portHandler, m_id, self.addr_goal_position, self.initial_positions[m_id])
        print("Mini Bento Ready.")
        return self

    def read_from_motor(self, motor_id):
        with self.lock: # Wait for our turn to use the COM port
            p, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, self.addr_present_position)
            v_raw, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, self.addr_present_velocity)
            l_raw, _, _ = self.packetHandler.read2ByteTxRx(self.portHandler, motor_id, self.addr_present_load)
        
        if p is None: return None, None, None
        return p, to_signed_32(v_raw), to_signed_16(l_raw)

    def set_goal_pos(self, motor_id, pos):
        with self.lock:
            self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, self.addr_goal_position, pos)

    def cycle_motor(self, motor_ID, pos1, pos2, wait_time, check_paused, check_running):
        while check_running():
            if check_paused():
                time.sleep(0.1)
                continue
            
            # Step 1: Move to Position 1 (Close)
            self.set_goal_pos(motor_ID, pos1)
            self._safe_sleep(wait_time, check_paused, check_running)

            # Step 2: Move to Position 2 (Open)
            if not check_paused() and check_running():
                self.set_goal(motor_ID, pos2)
                self._safe_sleep(wait_time, check_paused, check_running)

    def _safe_sleep(self, duration, check_paused, check_running):
        # Helper to allow interrupting a long wait time if we pause
        end_time = time.time() + duration
        while time.time() < end_time:
            if check_paused() or not check_running():
                break
            time.sleep(0.1)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Disable Torque and close port
        print("\nSafety Shutdown: Disabling Torque...")
        for m_id in self.motor_ids:
            with self.lock:
                self.packetHandler.write1ByteTxRx(self.portHandler, m_id, self.addr_torque_enable, 0)
        self.portHandler.closePort()
        print("Communication Closed.")