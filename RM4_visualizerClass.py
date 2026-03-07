'''
Visualizer class for CMPUT 605 Robot Module 4
[Insert description]
Written by: Quinn Boser, with assistance from Google Gemini 
March 2026
'''

import matplotlib.pyplot as plt
from collections import deque
import numpy as np

class ACVisualizer:
    def __init__(self, window_size=200, goal_pos=None):
        self.window_size = window_size
        self.goal_pos = goal_pos
        
        # Initialize deques to hold history
        self.pos_hist = deque([0] * window_size, maxlen=window_size)
        self.reward_hist = deque([0] * window_size, maxlen=window_size)

        # Setup the figure and subplots
        plt.ion()
        # self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 8))
        self.fig, self.axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Top Plot: Position
        self.line_pos, = self.axs[0].plot(list(self.pos_hist), color='blue', label='Position')
        self.axs[0].set_ylabel('Position', color='blue')
        if self.goal_pos is not None:
            self.axs[0].axhline(y=goal_pos, color='red', linestyle='--', label='Secret Goal Position')
        self.axs[0].legend(loc='upper right')

        # Bottom Plot: Reward
        self.line_reward, = self.axs[1].plot(list(self.reward_hist), color='green', label='Reward')
        self.axs[1].set_ylabel('Reward', color='green')

        self.fig.tight_layout()

    def update_data(self, pos, reward):
        # Add new data points to the histories
        self.pos_hist.append(pos)
        self.reward_hist.append(reward)

    def draw(self):
        # Refresh the plot lines
        self.line_pos.set_ydata(list(self.pos_hist))
        self.line_reward.set_ydata(list(self.reward_hist))
        
        # Update Y-limits dynamically
        self._smart_limit(self.axs[0], [self.pos_hist, [self.goal_pos]] if self.goal_pos is not None else [self.pos_hist])
        self._smart_limit(self.axs[1], [self.reward_hist])

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