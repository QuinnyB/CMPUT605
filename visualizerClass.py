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

        # Setup the figure and subplots
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        
        # Top Plot: Position and Velocity (TwinX)
        self.line_pos, = self.axs[0].plot(list(self.pos_hist), color='blue', label='Position')
        self.axs[0].set_ylabel('Position', color='blue')
        self.axs[0].set_ylim(0, 4095)
        
        self.ax0_right = self.axs[0].twinx()
        self.line_vel, = self.ax0_right.plot(list(self.vel_hist), color='green', alpha=0.6, label='Velocity')
        self.ax0_right.set_ylabel('Velocity', color='green')
        self.ax0_right.set_ylim(-100, 100) 

        # Middle Plot: Load
        self.line_load, = self.axs[1].plot(list(self.load_hist), color='red', alpha=0.6)
        self.axs[1].set_ylabel('Raw Load', color='red')
        self.axs[1].set_ylim(-400, 400)

        # Bottom Plot: Learning Signals
        self.line_c, = self.axs[2].plot(list(self.cumulant_hist), color='orange', alpha=0.6, label='Cumulant')
        self.line_pred, = self.axs[2].plot(list(self.pred_hist), color='purple', alpha=0.6, label='Prediction')
        self.line_verifier, = self.axs[2].plot(list(self.verifier_hist), color='brown', alpha=0.6, label='Verifier')
        self.axs[2].set_ylabel('Learning Signals', color='purple')
        self.axs[2].set_ylim(-0.5, 1.5)
        self.axs[2].legend(loc='upper right', fontsize='small')

        self.fig.tight_layout()

    def update_data(self, pos, vel, load, cumulant, pred, verifier=np.nan):
        # Add new data points to the histories
        self.pos_hist.append(pos)
        self.vel_hist.append(vel)
        self.load_hist.append(load)
        self.cumulant_hist.append(cumulant)
        self.pred_hist.append(pred)
        # self.verifier_hist.append(verifier)

    def update_verifier(self, verifier_buffer_length, gamma):
        self.verifier_hist.append(np.nan)
        verifier_indices = np.arange(verifier_buffer_length)    # Create an array of indices [0, 1, 2, ..., verifier_buffer_length-1]
        # Compute expected prediction for verifier_buffer_length steps in the past:
        verifier_buffer = list(self.cumulant_hist)[-verifier_buffer_length:]
        expected_pred = np.sum(verifier_buffer * (gamma ** verifier_indices)) * (1-gamma)
        self.verifier_hist[-(verifier_buffer_length+1)] = expected_pred

    def draw(self):
        # Refresh the plot lines
        self.line_pos.set_ydata(list(self.pos_hist))
        self.line_vel.set_ydata(list(self.vel_hist))
        self.line_load.set_ydata(list(self.load_hist))
        self.line_c.set_ydata(list(self.cumulant_hist))
        self.line_pred.set_ydata(list(self.pred_hist))
        self.line_verifier.set_ydata(list(self.verifier_hist))
        
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