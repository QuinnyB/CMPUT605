import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class RLVisualizer:
    def __init__(self, window_size=200):
        self.window_size = window_size
        
        # Initialize deques to hold history
        self.pos_hist = deque([0] * window_size, maxlen=window_size)
        self.vel_hist = deque([0] * window_size, maxlen=window_size)
        self.load_hist = deque([0] * window_size, maxlen=window_size)
        self.cumulant_hist = deque([0] * window_size, maxlen=window_size)
        self.pred_hist = deque([0] * window_size, maxlen=window_size)
        self.verifier_hist = deque([np.nan] * window_size, maxlen=window_size)
        self.bin_hist = deque([np.nan] * window_size, maxlen=window_size)  # For feature index (if desired)

        # Setup the figure and subplots
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        
        # Top Plot: Position and Velocity (TwinX)
        self.line_pos, = self.axs[0].plot(list(self.pos_hist), color='blue', label='Position')
        self.axs[0].set_ylabel('Position', color='blue')
        # self.axs[0].set_ylim(0, 4095)
        
        self.ax0_right = self.axs[0].twinx()
        self.line_vel, = self.ax0_right.plot(list(self.vel_hist), color='green', alpha=0.6, label='Velocity')
        self.ax0_right.set_ylabel('Velocity', color='green')
        # self.ax0_right.set_ylim(-100, 100) 

        # Middle Plot: Load
        self.line_load, = self.axs[1].plot(list(self.load_hist), color='red', alpha=0.6)
        self.axs[1].set_ylabel('Raw Load', color='red')
        # self.axs[1].set_ylim(-400, 400)

        # Bottom Plot: Learning Signals
        self.line_c, = self.axs[2].plot(list(self.cumulant_hist), color='orange', alpha=0.6, label='Cumulant')
        self.line_pred, = self.axs[2].plot(list(self.pred_hist), color='purple', alpha=0.6, drawstyle='steps-post', label='Prediction')
        self.line_verifier, = self.axs[2].plot(list(self.verifier_hist), color='brown', alpha=0.6, label='Verifier')
        self.ax0_right
        # Plot feature index on the same plot with a secondary y-axis
        self.ax2_right = self.axs[2].twinx()
        self.line_bin, = self.ax2_right.plot(list(self.bin_hist), color='blue', alpha=0.6, drawstyle='steps-post', label='Feature Index')
        self.ax2_right.set_ylabel('Feature Index', color='blue')
        self.axs[2].legend(loc='upper left', fontsize='small')

        self.fig.tight_layout()

    def update_data(self, pos, vel, load, cumulant, pred, feature_idx=None):
        # Add new data points to the histories
        self.pos_hist.append(pos)
        self.vel_hist.append(vel)
        self.load_hist.append(load)
        self.cumulant_hist.append(cumulant)
        self.pred_hist.append(pred) 
        if feature_idx is not None:
            self.bin_hist.append(feature_idx)
        else:
            self.bin_hist.append(np.nan)
        
    def update_verifier(self, expected_pred, idx_back):
        self.verifier_hist.append(np.nan)
        self.verifier_hist[-(idx_back)] = expected_pred

    def draw(self):
        # Refresh the plot lines
        self.line_pos.set_ydata(list(self.pos_hist))
        self.line_vel.set_ydata(list(self.vel_hist))
        self.line_load.set_ydata(list(self.load_hist))
        self.line_c.set_ydata(list(self.cumulant_hist))
        self.line_pred.set_ydata(list(self.pred_hist))
        self.line_verifier.set_ydata(list(self.verifier_hist))
        self.line_bin.set_ydata(list(self.bin_hist))

        # Update Y-limits dynamically
        self._smart_limit(self.axs[0], [self.pos_hist])
        self._smart_limit(self.ax0_right, [self.vel_hist])
        self._smart_limit(self.axs[1], [self.load_hist])
        self._smart_limit(self.axs[2], [self.cumulant_hist, self.pred_hist, self.verifier_hist])
        self._smart_limit(self.ax2_right, [self.bin_hist])
    
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def process_events(self):
        # Keep the window alive without adding new data
        if self.is_open():
            # flush_events handles the 'X' button and resizing
            self.fig.canvas.flush_events()
            # This is a non-blocking way to keep the UI responsive
            self.fig.canvas.start_event_loop(0.001)

    def is_open(self):
        # Check if the window is still open
        return plt.fignum_exists(self.fig.number)
    
    def _smart_limit(self, ax, data_lists, padding=0.1, default_range=(-1, 1)):
        # Combine all data and filter out NaNs (critical for the verifier)
        combined_data = []
        for d in data_lists:
            combined_data.extend([x for x in d if not np.isnan(x)])     
        if not combined_data:
            ax.set_ylim(default_range)
            return
        d_min, d_max = min(combined_data), max(combined_data)
        # If all data is the same value (like all zeros at start), use default range
        if d_min == d_max:
            ax.set_ylim(d_min + default_range[0], d_min + default_range[1])
        else:
            margin = (d_max - d_min) * padding
            ax.set_ylim(d_min - margin, d_max + margin)