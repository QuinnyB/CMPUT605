'''
Helper functions for CMPUT 605 Robot Module 4
Written by: Quinn Boser, with assistance from Google Gemini 
March 2026
'''

from typing import List

import matplotlib.pyplot as plt
from collections import deque
from dynamixel_sdk import *
import math
import numpy as np
from pynput import keyboard

# --- Function Definitions ----------------------------------------------------------------
# Softmax probability:
def compute_softmax_prob(actor_w, active_indices):
    # Input: actor_w is a 2D array of shape (feature_vector_length, num_actions)
    #        active_indices is a list of indices representing the active features
    # Output: a 1D array of shape (num_actions) representing the softmax probabilities of each action
    # Compute the preferences for each action in the current state:
    preferences = actor_w[active_indices, :].sum(axis=0)
    # Calculate constant c for numerical stability (max of preferences):
    c = np.max(preferences)
    # Calculate the softmax probabilities:
    numerator = np.exp(preferences - c)
    denominator = np.sum(numerator) 
    return numerator / denominator

# Softmax action selection:
def get_softmax_action(softmax_probs):
    # Input: softmax_probs is a 1D array of shape (num_actions) representing the softmax probabilities of each action
    # Output: an integer representing the index of the selected action
    # Select an action based on the softmax probabilities using np.random.choice
    num_actions = len(softmax_probs)
    action = np.random.choice(num_actions, p=softmax_probs)
    return action

# Vector summation to calculate phasor magnitude and angle:
def get_phasor(signals):
    # Input: signals is a 1D array of length n representing the MAV of each EMG channel
    # Output: magnitude and angle (in degrees) of the resulting phasor
    n = len(signals)
    # Generate angles for each signal (0 to 360 degrees, assumes evenly spaced)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Decompose each signal into its X and Y components, then sum the components
    x_total = np.sum(signals * np.cos(angles))
    y_total = np.sum(signals * np.sin(angles))
    # Calculate Magnitude and Angle
    magnitude = np.sqrt(x_total**2 + y_total**2)
    angle_rad = np.arctan2(y_total, x_total)
    # Convert angle to degrees for easier reading
    angle_deg = np.degrees(angle_rad)
    return magnitude, angle_deg

# Signed Conversions:
def to_signed_32(val):
    if val > 2147483647 :  # If greater than (2^31 - 1)
        return val - 4294967296  # Subtract 2^32
    else:   return val
    
def to_signed_16(val):
    if val > 32767:  # If greater than (2^15 - 1)
        return val - 65536  # Subtract 2^16
    else:   return val

# Normalize function
def normalize(value, min_val, max_val):
    value = np.array(value)
    min_val = np.array(min_val)
    max_val = np.array(max_val)
    np.clip(value, min_val, max_val, out=value)
    return (value - min_val) / (max_val - min_val)

# Binning function - expects normalized value between 0 and 1
def bin(value, num_bins):
    bin_index = math.ceil(value * num_bins) - 1
    if bin_index < 0:
        bin_index = 0
    if bin_index >= num_bins:
        bin_index = num_bins - 1
    return bin_index

# Generic one-hot featurization function for multiple signals:
def featurize_grid(signals, ranges, bin_counts):
    # Input:
    #   signals:    List of values [s1, s2, ...]
    #   ranges:     List of (min, max) tuples [(min1, max1), ...]
    #   bin_counts: List of integers [bins1, bins2, ...]
    indices = [] 
    # Calculate discrete bin index for each signal
    for val, (s_min, s_max), num_bins in zip(signals, ranges, bin_counts):
        # Normalize signal to [0, 1]
        val_norm = normalize(val, s_min, s_max)
        indices.append(bin(val_norm, num_bins)) 
    # Calculate the flattened 1D index
    # Logic: index = i0*(bins1*bins2*...) + i1*(bins2*...) + ... +  in
    flat_idx = 0
    multiplier = 1
    for i in reversed(range(len(indices))):
        flat_idx += indices[i] * multiplier
        multiplier *= bin_counts[i]    
    # Create the vector
    total_size = np.prod(bin_counts)
    x = np.zeros(total_size, dtype=int)
    x[flat_idx] = 1 
    return x, flat_idx, indices

# Function to check if the sampled action matches the physical keyboard input
def check_action_match(action_index, pressed_key, action_key_map):
    # Input:
    #     action_index (int): The index (0, 1, 2) from the learner.
    #     pressed_key (str): The key currently held ('a', 's', 'd') or None.
    #     action_key_map (dict): The mapping dictionary.    
    # Returns:
    #     bool: True if they match, False otherwise.
    # If no key is pressed, it's automatically not a match
    if pressed_key is None:
        return False
    # Get a list of keys from the dict: ['a', 's', 'd']
    keys = list(action_key_map.keys())
    # Safety check for index out of bounds
    if action_index < 0 or action_index >= len(keys):
        return False
    # Check if the key at that index matches the pressed key
    return keys[action_index] == pressed_key

# Convert quaternion to rotation matrix:
def quat_to_mat(w, x, y, z):
    # Check that the input is normalized
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm != 1.0:
        w /= norm
        x /= norm
        y /= norm
        z /= norm
    # Standard formula for conversion
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])

# Convert rotation matrix to ZXY moving axes angles:
def mat_to_ZXY(mat):
    r11, r12, r13 = mat[0]
    r21, r22, r23 = mat[1]
    r31, r32, r33 = mat[2]
    x_angle = math.atan2(r32, math.sqrt(r31**2 + r33**2))
    cosx = math.cos(x_angle)
    y_angle = math.atan2(r31/cosx, r33/cosx)
    z_angle = math.atan2(r12/cosx, r22/cosx)
    return np.degrees([z_angle, x_angle, y_angle])