import numpy as np
from helperFunctions import normalize

class TileCoder:
    def __init__(self, coder_dict):
        """
        Assume coder_dist contains:
        {
            'num_tilings': int
            'ranges':     List of (min, max) tuples [(min1, max1), ...]
            'bins_per_dim': list of ints (length = num_dims)
            'use_hashing': bool (optional, default False),
            'hash_size': int (optional, default 4096)
        }
        """
        # Set parameters from the dictionary
        self.num_tilings = coder_dict['num_tilings']
        self.low = np.array([r[0] for r in coder_dict['ranges']])
        self.high = np.array([r[1] for r in coder_dict['ranges']])
        self.bins_per_dim = np.array(coder_dict['bins_per_dim'])
        self.num_dims = len(self.bins_per_dim)
        self.wrap_dims = coder_dict.get('wrap_dims', [False] * self.num_dims)
        self.use_hashing = coder_dict.get('use_hashing', False) # Default to False if missing
        self.hash_size = coder_dict.get('hash_size', 4096) 

        # Calculate derived parameters
        self.tiles_per_tiling = np.prod(self.bins_per_dim)
        if self.use_hashing:
            self.total_size = self.hash_size
        else:
            self.total_size = int(self.tiles_per_tiling * self.num_tilings)
        
        # Calculate asymmetrical offsets
        # Generate the first 'num_dims' odd integers: [1, 3, 5, ...] 
        odd_ints = np.arange(1, 2 * self.num_dims, 2) 
        # Column vector of tiling indices:[0, 1, ..., num_tilings-1]'
        tiling_indices = np.arange(self.num_tilings)[:, np.newaxis]
        # Create offsets: shape (num_tilings, num_dims), where each tiling is shifted by a different amount in each dimension
        self.offsets = (tiling_indices * odd_ints) / self.num_tilings

    def get_indices(self, state):
        norm_state = normalize(state, self.low, self.high)
        # print(f"Normalized state: {norm_state}")
        indices = []
        
        for i in range(self.num_tilings):
            # Scale the [0,1] state to the number of bins
            # Add the fractional offset for this tiling
            scaled_state = (norm_state * self.bins_per_dim) + self.offsets[i]
            
            # Initialize an empty array for the processed coordinates
            coords = np.zeros(self.num_dims, dtype=int)
        
            for d in range(self.num_dims):
                # Floor to integer
                val = int(np.floor(scaled_state[d]))
            
                if self.wrap_dims[d]:
                    # Periodic: wrap around using modulo
                    coords[d] = val % self.bins_per_dim[d]
                else:
                    # Bounded: clamp between 0 and bins-1
                    coords[d] = np.clip(val, 0, self.bins_per_dim[d] - 1)
                
            if self.use_hashing:
                idx = hash(tuple(coords) + (i,)) % self.hash_size
            else:
                flat_coords = np.ravel_multi_index(coords, self.bins_per_dim)
                # print(f"Tiling {i}: Flat Coords={flat_coords}")
                idx = (i * self.tiles_per_tiling) + flat_coords
            
            indices.append(int(idx))

        return indices