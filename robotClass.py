'''
Class definition for Mini Bento robot arm (https://github.com/BLINCdev/Mini-Bento)
Allows setting goal position of a given motor, reading from a given motor, 
and setting a behaviour policy for a motor (either cycling between two positions or random movement)
Written by: Quinn Boser, with assistance from Google Gemini 
Feb. 2026
'''

import threading
import time
import math
import random
from dynamixel_sdk import *
from helperFunctions import to_signed_32, to_signed_16 

class MiniBento:
    def __init__(self, port_name, baudrate, motor_velo, iniital_positions):
        self.portHandler = PortHandler(port_name)
        self.baudrate = baudrate
        self.motor_velo = motor_velo
        self.initial_positions = iniital_positions
        self.packetHandler = PacketHandler(2.0) # Using protocol version 2.0
        self.motor_ids = [1, 2, 4, 5]           # Motor IDs for Mini Bento
        self.lock = threading.Lock() 
        # Set motor addresses:
        # (see https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)
        self.addr_torque_enable = 64
        self.addr_goal_position = 116
        self.addr_profile_velocity = 112
        self.addr_present_load = 126
        self.addr_present_velocity = 128
        self.addr_present_position = 132
        
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
        # Read current position, velocity, and load (current draw) from motor
        with self.lock:     # Wait our turn to use the COM port
            p, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, self.addr_present_position)
            v_raw, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, self.addr_present_velocity)
            l_raw, _, _ = self.packetHandler.read2ByteTxRx(self.portHandler, motor_id, self.addr_present_load)
        if p is None: return None, None, None
        return p, to_signed_32(v_raw), to_signed_16(l_raw)

    def set_goal_pos(self, motor_id, pos):
        # Set a goal position for motor
        with self.lock:     # Wait our turn to use the COM port
            self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, self.addr_goal_position, pos)

    def cycle_motor(self, motor_ID, pos1, pos2, wait_time, check_paused, check_running):
        # Move motor between two goal positions repeatedly
        while check_running():
            if check_paused():
                time.sleep(0.1)
                continue
            # Move to Position 1
            self.set_goal_pos(motor_ID, pos1)
            self._safe_sleep(wait_time, check_paused, check_running) # (wait to get to position)
            # Move to Position 2
            if not check_paused() and check_running():
                self.set_goal_pos(motor_ID, pos2)
                self._safe_sleep(wait_time, check_paused, check_running) # (wait to get to position)

    def random_walk(self, motor_ID, lim1, lim2, check_paused, check_running):
        # Repeatedly move motor to randomly chosen positions within given limits
        last_pos, _ , _ = self.read_from_motor(motor_ID)
        while check_running():
            if check_paused():
                time.sleep(0.1)
                continue
            # Generate random position within limits
            pos = random.randint(lim1, lim2)
            # Calculate time required to move from last  goal position to new position
            wait_time = math.ceil((abs(pos - last_pos) / (4096 * 0.229 * self.motor_velo)) * 60)
            # Set new goal position, update last position, and wait to get to position
            self.set_goal_pos(motor_ID, pos)
            last_pos = pos
            self._safe_sleep(wait_time, check_paused, check_running)    
            
    def _safe_sleep(self, duration, check_paused, check_running):
        # Helper to allow interrupting a long wait time if pause button (spacebar) pressed
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