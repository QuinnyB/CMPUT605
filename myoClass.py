'''
Class definitions for Myo Armband data acquisition and processing
Allows reading of raw 8-channel EMG data and IMU (Orientation, Accel, Gyro)
Written by: Quinn Boser, with assistance from Google Gemini 
Mar. 2026
'''

import threading
import time
import numpy as np
from pyomyo import Myo, emg_mode
from scipy.signal import iirnotch, lfilter

# --- Myo Armband Class -----------------------------------------------------------------------------
class MyoArmband:
    def __init__(self, mode=emg_mode.RAW):
        self.mode = mode
        self.myo = None
        self.hub_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Internal storage for the latest data frames
        self.latest_emg = [0] * 8
        self.latest_quat = [0] * 4
        self.latest_accel = [0] * 3
        self.latest_gyro = [0] * 3

    def __enter__(self):
        try:
            # Initialize the Myo
            self.myo = Myo(mode=self.mode)
            
            # Register handlers to update internal storage
            self.myo.add_emg_handler(self._emg_callback)
            self.myo.add_imu_handler(self._imu_callback)
            
            self.myo.connect()
            self.running = True
            
            # Start the background thread (similar to your robot worker)
            self.hub_thread = threading.Thread(target=self._run_loop, daemon=True)
            self.hub_thread.start()
            
            print(f"Myo Armband connected.")
            return self
        except Exception as e:
            print(f"Failed to connect to Myo: {e}")
            raise

    def _run_loop(self):
        """Background thread to process Bluetooth packets."""
        while self.running:
            # NOTICE: No lock here! 
            # We let the internal pyomyo logic handle the radio.
            try:
                self.myo.run() 
            except Exception as e:
                print(f"Myo Run Error: {e}")
            time.sleep(0.001)

    def _emg_callback(self, emg, worker):
        # We ONLY lock when writing the data to the shared variable
        with self.lock:
            self.latest_emg = list(emg)

    def _imu_callback(self, quat, acc, gyro):
        # We ONLY lock when writing the data to the shared variable
        with self.lock:
            self.latest_quat = list(quat)
            self.latest_accel = list(acc)
            self.latest_gyro = list(gyro)

    def get_data(self):
        """Returns a snapshot of all current sensor data."""
        with self.lock:
            return {
                "emg": self.latest_emg,
                "quat": self.latest_quat,
                "accel": self.latest_accel,
                "gyro": self.latest_gyro
            }

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.hub_thread:
            self.hub_thread.join(timeout=1.0)
        if self.myo:
            self.myo.disconnect()
        print("Myo Armband Disconnected.")

# --- EMG Data Processor Class -----------------------------------------------------------------------------
class EMGProcessor:
    def __init__(self, num_channels=8, window_size=20, fs=200):
        self.num_channels = num_channels
        self.window_size = window_size
        self.fs = fs  # Sample rate in Hz
        
        # Design a Notch Filter for 60Hz powerline noise
        # Quality factor (Q) determines how narrow the 'cut' is
        self.b, self.a = iirnotch(60.0, 30.0, fs)
        
        # Dynamically create buffers for the specified number of channels
        self.buffers = [[] for _ in range(self.num_channels)]

    def process_frame(self, raw_emg_list):
        """
        Takes a list of raw EMG values and returns the MAV for each.
        Works for any number of channels defined at initialization.
        """
        if len(raw_emg_list) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, but got {len(raw_emg_list)}")

        processed_mav = []
        
        for i in range(self.num_channels):
            # 1. Update buffer
            self.buffers[i].append(raw_emg_list[i])
            
            # Keep buffer size manageable (2x window size for filter stability)
            if len(self.buffers[i]) > self.window_size * 2:
                self.buffers[i].pop(0)
            
            # 2. Apply 60Hz Notch Filter
            # Note: lfilter returns an array; we take the last window_size elements
            filtered_data = lfilter(self.b, self.a, self.buffers[i])
            
            # 3. Calculate Mean Absolute Value (MAV)
            # MAV = (1/N) * Σ |x_i|
            window = filtered_data[-self.window_size:]
            mav = np.mean(np.abs(window))
            processed_mav.append(mav)
            
        return processed_mav