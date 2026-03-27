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
def compute_softmax_prob(actor_w, state_x):
    # Input: actor_w is a 2D array of shape (feature_vector_length, num_actions)
    #        state_x is a 1D array of shape (feature_vector_length)
    # Output: a 1D array of shape (num_actions) representing the softmax probabilities of each action
    # Compute the preferences for each action in the current state:
    preferences = state_x @ actor_w  
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
    if value < min_val:
        value = min_val
    if value > max_val:
        value = max_val
    return (value - min_val) / (max_val - min_val)

# Binning function - expects normalized value between 0 and 1
def bin(value, num_bins):
    bin_index = math.ceil(value * num_bins) - 1
    if bin_index < 0:
        bin_index = 0
    if bin_index >= num_bins:
        bin_index = num_bins - 1
    return bin_index

def featurize_grid(signals, ranges, bin_counts):
    # Input:
    #   signals:    List of values [s1, s2, ...]
    #   ranges:     List of (min, max) tuples [(min1, max1), ...]
    #   bin_counts: List of integers [bins1, bins2, ...]
    indices = [] 
    # 1. Calculate discrete bin index for each signal
    for val, (s_min, s_max), num_bins in zip(signals, ranges, bin_counts):
        # Normalize signal to [0, 1]
        val_norm = normalize(val, s_min, s_max)
        indices.append(bin(val_norm, num_bins)) 
    # 2. Calculate the flattened 1D index
    # Logic: index = i1*(bins2*bins3...) + i2*(bins3...) + ... + in
    flat_idx = 0
    multiplier = 1
    for i in reversed(range(len(indices))):
        flat_idx += indices[i] * multiplier
        multiplier *= bin_counts[i]    
    # 3. Create the sparse vector
    total_size = np.prod(bin_counts)
    x = np.zeros(total_size, dtype=int)
    x[flat_idx] = 1 
    return x, flat_idx, indices