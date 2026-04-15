
from tileCoderClass import TileCoder
from helperFunctions import featurize_grid
import numpy as np

EMG_MAG_RANGE = (0, 100)   # Expected range of EMG phasor magnitude (for featurization)
EMG_ANGLE_RANGE = (-180, 180)  # Expected range of EMG phasor angle (for featurization)
SHO_LIMS = [1500, 2500]     # Shoulder motor limits (encoder values [CCW, CW])

config = {
    'num_tilings': 1,
    'ranges': [EMG_MAG_RANGE, EMG_ANGLE_RANGE, SHO_LIMS],  # For featurization
    'bins_per_dim': [3, 8, 2],  # As requested
    'use_hashing': False        # Small space (3*8*2*10 = 480 weights), no hash needed
}

# 3. Initialize TileCoder
# Assuming you named your class TileCoder and it's in scope
tc = TileCoder(config)

# 4. Test with a sample state
# Let's pick a state right in the middle: [50, 0, 2000]
sample = [50, 170, 1700]

active_indices = tc.get_indices(sample)
# print(f"Active Indices (one per tiling): {active_indices}")
# print(f"Total Weights in Model: {tc.total_size}")

x, flat_idx, indices = featurize_grid(sample, config['ranges'], config['bins_per_dim'])
# print(f"Featurize Grid Output: flat_idx={flat_idx}, indices={indices}")


x_h = x_h = np.eye(3)[1]
print(f"x_h: {x_h}")