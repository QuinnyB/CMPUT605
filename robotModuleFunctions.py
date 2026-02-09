"""Utility functions for robot control and learning."""
import math
import numpy as np


# --- Signed Value Conversions -------------------------------------------------------
def to_signed_32(val):
    """Convert unsigned 32-bit integer to signed."""
    if val > 2147483647:  # If greater than (2^31 - 1)
        return val - 4294967296  # Subtract 2^32
    else:
        return val


def to_signed_16(val):
    """Convert unsigned 16-bit integer to signed."""
    if val > 32767:  # If greater than (2^15 - 1)
        return val - 65536  # Subtract 2^16
    else:
        return val


# --- Normalization and Binning -------------------------------------------------------
def normalize(value, min_val, max_val):
    """
    Normalize a value to the range [0, 1].
    
    Args:
        value: Value to normalize
        min_val: Minimum value of the range
        max_val: Maximum value of the range
        
    Returns:
        float: Normalized value in [0, 1]
    """
    # Clamp value to range
    if value < min_val:
        value = min_val
    if value > max_val:
        value = max_val
    
    return (value - min_val) / (max_val - min_val)


def bin(value, num_bins):
    """
    Assign a normalized value to a bin index.
    
    Args:
        value: Normalized value in [0, 1]
        num_bins: Number of bins to quantize into
        
    Returns:
        int: Bin index in range [0, num_bins-1]
    """
    bin_index = math.ceil(value * num_bins) - 1
    
    # Clamp to valid range
    if bin_index < 0:
        bin_index = 0
    if bin_index >= num_bins:
        bin_index = num_bins - 1
    
    return bin_index

