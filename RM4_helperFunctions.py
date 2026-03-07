'''
Helper functions for CMPUT 605 Robot Module 4
Written by: Quinn Boser, with assistance from Google Gemini 
March 2026
'''

import matplotlib.pyplot as plt
from collections import deque
from dynamixel_sdk import *
import math
import numpy as np
from pynput import keyboard

# --- Function Definitions ----------------------------------------------------------------
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

def get_softmax_action(softmax_probs):
    # Input: softmax_probs is a 1D array of shape (num_actions) representing the softmax probabilities of each action
    # Output: an integer representing the index of the selected action
    # Select an action based on the softmax probabilities using np.random.choice
    num_actions = len(softmax_probs)
    action = np.random.choice(num_actions, p=softmax_probs)
    return action

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

# Convert position and velocity into feature vector X
def featurize_pos(pos, pos1, pos2, num_pos_bins):
    # Normalize position and velocity to [0, 1]
    pos_norm = normalize(pos, pos1, pos2)
    # Determine bin indices
    pos_bin = bin(pos_norm, num_pos_bins)
    # Create feature vector
    x = np.zeros(num_pos_bins, dtype=int)
    x[pos_bin] = 1
    assert x.sum() == 1, "Feature vector should have exactly one active feature"
    return x, pos_bin
