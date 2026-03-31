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
    def __init__(self, window_size = 40):
        self.mode = emg_mode.RAW
        self.myo = None
        self.hub_thread = None
        self.running = False
        self.lock = threading.Lock()
        self.window_size = window_size

        # Buffers for the background thread to use (8 channels)
        self.raw_emg_buffers = [[] for _ in range(8)]
        # Internal storage for the latest data frames
        self.latest_emg_mav = [0.0] * 8
        self.latest_quat = [0] * 4
        self.latest_accel = [0] * 3
        self.latest_gyro = [0] * 3

    def __enter__(self):
        try:
            # Initialize the Myo
            self.myo = Myo(mode=self.mode)
            # Register handlers to update internal storage
            self.myo.add_emg_handler(self._emg_worker_callback)
            self.myo.add_imu_handler(self._imu_worker_callback)
            # Connect to the Myo
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
            try:
                self.myo.run() 
            except Exception as e:
                print(f"Myo Run Error: {e}")
            time.sleep(0.001)

    def _emg_worker_callback(self, emg, _):
        # This is called every time a new EMG packet arrives (200 times per second).
        temp_mav = []
        for i in range(8):
            self.raw_emg_buffers[i].append(emg[i])
            # Maintain buffer for filtering/MAV
            if len(self.raw_emg_buffers[i]) > self.window_size * 2:
                self.raw_emg_buffers[i].pop(0)
            mav = np.mean(np.abs(self.raw_emg_buffers[i][-self.window_size:]))
            temp_mav.append(mav)
        with self.lock:
            self.latest_emg_mav = temp_mav

    def _imu_worker_callback(self, quat, acc, gyro):
        with self.lock:
            self.latest_quat = list(quat)
            self.latest_accel = list(acc)
            self.latest_gyro = list(gyro)

    def get_data(self):
        # Thread-safe access for slow main loop
        with self.lock:
            return self.latest_emg_mav, self.latest_quat, self.latest_accel, self.latest_gyro

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.hub_thread:
            self.hub_thread.join(timeout=1.0)
        if self.myo:
            self.myo.disconnect()
        print("Myo Armband Disconnected.")

