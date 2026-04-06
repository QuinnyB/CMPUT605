'''
Visualizer class for CMPUT 605 Final Project
Written by: Quinn Boser, with assistance from Google Gemini
April 2026
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.stats import norm
from matplotlib.patches import Patch

class ACVisualizer:
    def __init__(self, visualizer_dict = {}):
        """
        Assumes visualizer_dict contains:
        {
            "window_size": int, Number of recent data points to display
            "max_EMG": float or None, The maximum expected EMG magnitude for scaling the polar plot
            "reward_range": tuple or None, The min/max reward values for the Y-axis (e.g., (-1, 0))
            "action_mode": str, 'discrete' or 'continuous' - whether the action plot should show discrete probabilities or a continuous distribution
            "action_key_map": dict, Mapping of keys to action names for discrete mode (e.g., {'a': 'Close', 's': 'Rest', 'd': 'Open'})
            "action_range": tuple or None, The min/max action values for the continuous X-axis.
        }
        """
        self.window_size = visualizer_dict.get("window_size", 100)
        self.max_EMG = visualizer_dict.get("max_EMG", 100)
        self.show_feature_idx = visualizer_dict.get("show_feature_idx", False)
        self.reward_range = visualizer_dict.get("reward_range")
        self.mode = visualizer_dict.get("action_mode")
        self.action_key_map = visualizer_dict.get("action_key_map")
        self.action_labels = list(self.action_key_map.values())
        self.action_range = visualizer_dict.get("action_range", [-3,3])

        # Disable Matplotlib's internal key catchers so they don't interfere with keyPressHandler
        plt.rcParams['keymap.save'] = []
        plt.rcParams['keymap.quit'] = []
        plt.rcParams['keymap.fullscreen'] = []

        # History buffers
        self.reward_hist = deque([np.nan] * self.window_size, maxlen=self.window_size)
        self.avg_reward_hist = deque([np.nan] * self.window_size, maxlen=self.window_size)

        # Tracking variables
        self.current_action_data = None
        self.sampled_action = None # To store the specific action taken
        self.step_count = 0
        self.avg_step_time = 0.0
        self.start_time = time.perf_counter() # Initialize timer

        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 2)

        # Top Left: Keyboard Indicator
        self.ax_key = self.fig.add_subplot(gs[0, 0])
        self.ax_key.set_xticks([])
        self.ax_key.set_yticks([])
        self.ax_key.set_facecolor('#f0f0f0') # Light grey background for the status box
        # Large centered text for Key and Action
        self.text_key = self.ax_key.text(0.5, 0.65, "Use Keyboard to Indicate Intended Action", 
                                        ha='center', va='center', fontsize=14, fontweight='bold')
        self.text_action = self.ax_key.text(0.5, 0.35, " | ".join([f"'{k}' = {v}" for k, v in self.action_key_map.items()]), 
                                           ha='center', va='center', fontsize=12, color='blue', fontweight='bold')
        self.ax_key.set_title("User Keyboard Input Status")

        # Top Right: EMG Phasor
        self.ax_polar = self.fig.add_subplot(gs[0, 1], projection='polar')
        self.ax_polar.set_xticklabels([])   # Remove 0, 45, 90... degree labels
        self.ax_polar.grid(True, alpha=0.3) # Make grid lines lighter
        self.line_phasor, = self.ax_polar.plot([0, 0], [0, 0], color='red', lw=3, marker='o')
        self.ax_polar.set_ylim(0, self.max_EMG)
        self.ax_polar.set_title("EMG Phasor", pad=15)
        if self.show_feature_idx:
            self.text_feature = self.ax_polar.text(1.3, 0.0, "Feature: 0", 
                                                   ha='right', va='bottom', fontsize=12, 
                                                   color='darkred', transform=self.ax_polar.transAxes)
        
        # Bottom Left: Reward
        self.ax_reward = self.fig.add_subplot(gs[1, 0])
        self.line_rewardAvg, = self.ax_reward.plot(list(self.avg_reward_hist), color='lightgreen', alpha=0.9, label='Avg Reward')
        self.line_reward, = self.ax_reward.plot(list(self.reward_hist), color='green', label='Current Reward')
        
        self.ax_reward.set_xlim([0, self.window_size])
        self.ax_reward.set_xlabel('Time Steps', fontsize=12)
        self.ax_reward.set_ylim(self.reward_range)
        self.ax_reward.set_ylabel('Reward', fontsize=14)
        self.ax_reward.legend(loc='upper left')

        # Bottom Right: Action Distribution
        self.ax_probs = self.fig.add_subplot(gs[1, 1])
        if self.mode == 'discrete':
            num_actions = len(self.action_labels) if self.action_labels else 2
            self.bar_probs = self.ax_probs.bar(range(num_actions), [0]*num_actions, color='purple', alpha=0.6)
            self.ax_probs.set_ylim(0, 1.1)
            self.ax_probs.set_ylabel('Action Probability', fontsize=12)
            if self.action_labels:
                self.ax_probs.set_xticks(range(num_actions))
                self.ax_probs.set_xticklabels(self.action_labels)
            # Create custom legend for bars
            legend_elements = [
                Patch(facecolor='purple', alpha=0.6, label='Policy $\pi(a|s)$'),
                Patch(facecolor='orange', label='Sampled Action')
            ]
            self.ax_probs.legend(handles=legend_elements, loc='upper left')
        else:
            # Continuous mode: Plotting a Gaussian PDF
            self.x_eval = np.linspace(self.action_range[0], self.action_range[1], 200)
            self.line_dist, = self.ax_probs.plot(self.x_eval, [0]*200, color='purple', lw=2, label='Policy $\pi$')
            # Initialize a vertical line for the sampled action
            self.v_line = self.ax_probs.axvline(x=0, color='orange', linestyle='--', alpha=0.8, label='Sampled Action')
            self.ax_probs.set_xlim(self.action_range)
            self.ax_probs.set_xlabel('Position Adjustment Multiplier', fontsize=12)
            self.ax_probs.set_ylim(0, 1.0)
            self.ax_probs.set_ylabel('Probability Density', fontsize=14)
            self.ax_probs.legend(loc='upper left')

        # Adjust layout to make room for the internal suptitle
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def update_data(self, key, emg_mag, emg_angle, reward, avg_reward, action_data, sampled_action, feature_index=None):
        """
        Args:
            key: The character currently pressed (e.g., 'a', 's', 'd') or None
            emg_mag: The magnitude of the EMG phasor for scaling the polar plot
            emg_angle: The angle of the EMG phasor in degrees for the polar plot
            reward, avg_reward: Scalar values
            action_data: 
                - If discrete: List of probabilities [p1, p2, ...]
                - If continuous: Tuple of (mean, std_dev)
            sampled_action:
                - If discrete: Integer index of the action taken
                - If continuous: Float value of the action taken
            feature_index: Integer index of the current feature vector

        """
        self.emg_mag = emg_mag
        self.emg_angle = np.radians(emg_angle)
        self.reward_hist.append(reward)
        self.avg_reward_hist.append(avg_reward)
        self.current_key = key
        self.current_action_data = action_data
        self.sampled_action = sampled_action
        self.feature_index = feature_index
        self.step_count += 1
        elapsed = time.perf_counter() - self.start_time
        self.avg_step_time = self.avg_step_time + (elapsed - self.avg_step_time) / self.step_count
        self.start_time = time.perf_counter()  # Reset timer for next step

    def draw(self):
        if not self.is_open(): return

        # Update Global Header with Step Count and Step Time
        self.fig.suptitle(f"Training Progress | Total Steps: {self.step_count} | Avg Step Time: {self.avg_step_time*1000:.2f} ms", 
                          fontsize=14)

        # Update Top Plot (Key Display)
        if self.current_key in self.action_key_map:
            self.text_key.set_text(f"Key Pressed: '{self.current_key}'")
            self.text_action.set_text(f"Intended Action: {self.action_key_map[self.current_key]}")
            self.ax_key.set_facecolor('#e1f5fe') # Light blue when active
        else:
            self.text_key.set_text("Use Keyboard to Indicate Intended Action")
            self.text_action.set_text(" | ".join([f"'{k}' = {v}" for k, v in self.action_key_map.items()]))
            self.ax_key.set_facecolor('#f0f0f0')

        # Update EMG Polar Plot
        # set_xdata is the angle (theta) and set_ydata is the radius (r)
        self.line_phasor.set_xdata([0, self.emg_angle])
        self.line_phasor.set_ydata([0, self.emg_mag])
        # Update Feature Index Text
        if self.show_feature_idx:
            self.text_feature.set_text(f"Feature Index: {self.feature_index}")
        
        # Update Reward lines
        self.line_reward.set_ydata(list(self.reward_hist))
        self.line_rewardAvg.set_ydata(list(self.avg_reward_hist))
        
        # Update Action Plot
        if self.current_action_data is not None:
            if self.mode == 'discrete':
                for i, bar in enumerate(self.bar_probs):
                    bar.set_height(self.current_action_data[i])
                    # Highlight the sampled action in orange, others in purple
                    if i == self.sampled_action:
                        bar.set_color('orange')
                        bar.set_alpha(0.9)
                    else:
                        bar.set_color('purple')
                        bar.set_alpha(0.6)
            else:
                mu, sigma = self.current_action_data
                y_pdf = norm.pdf(self.x_eval, mu, max(sigma, 1e-5))
                self.line_dist.set_ydata(y_pdf)
                
                # Update vertical line position for sampled action
                self.v_line.set_xdata([self.sampled_action, self.sampled_action])
                # Dynamically scale Y-axis for the PDF
                self.ax_probs.set_ylim(0, max(y_pdf) * 1.2)

        if self.reward_range is None:
            self._smart_limit(self.ax_reward, [self.reward_hist, self.avg_reward_hist])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _smart_limit(self, ax, data_lists, padding=0.1, default_range=(-1, 1)):
        combined_data = []
        for d in data_lists:
            combined_data.extend([x for x in d if not np.isnan(x)])     
        if not combined_data:
            ax.set_ylim(default_range)
            return
        d_min, d_max = min(combined_data), max(combined_data)
        if d_min == d_max:
            ax.set_ylim(d_min + default_range[0], d_min + default_range[1])
        else:
            margin = (d_max - d_min) * padding
            ax.set_ylim(d_min - margin, d_max + margin)

    def is_open(self):
        return plt.fignum_exists(self.fig.number)

    def process_events(self):
        if self.is_open():
            self.fig.canvas.flush_events()
            self.fig.canvas.start_event_loop(0.001)