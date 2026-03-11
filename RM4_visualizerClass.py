'''
Visualizer class for CMPUT 605 Robot Module 4
Generates a real-time visualization with three subplots:
1) Scrolling Motor position (with optional horizontal line for goal position)
2) Scrolling Reward and average reward
3) Action distribution (discrete probabilities or continuous Gaussian)
Note to self: matplotlib named colours can be found here: https://matplotlib.org/stable/gallery/color/named_colors.html
Written by: Quinn Boser, with assistance from Google Gemini 
March 2026
'''

import numpy as np
from collections import deque
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 

class ACVisualizer:
    def __init__(self, visualizer_dict = {}):
        """
        Assumes visualizer_dict contains:
        {
            "window_size": int, Number of recent data points to display
            "goal_pos": float or None, Optional horizontal line to indicate goal position on the plot
            "pos_range": tuple or None, The min/max position values for the Y-axis (e.g., (0, 4095))
            "reward_range": tuple or None, The min/max reward values for the Y-axis (e.g., (-1, 0))
            "action_mode": str, 'discrete' or 'continuous' - whether the action plot should show discrete probabilities or a continuous distribution
            "action_labels": list or None, List of action names for discrete mode (e.g., ['Left', 'Right'])
            "action_range": tuple or None, The min/max action values for the continuous X-axis.
        }
        """
        self.window_size = visualizer_dict.get("window_size")
        self.goal_pos = visualizer_dict.get("goal_pos")
        self.pos_range = visualizer_dict.get("pos_range")
        self.reward_range = visualizer_dict.get("reward_range")
        self.mode = visualizer_dict.get("action_mode")
        self.action_labels = visualizer_dict.get("action_labels")
        self.action_range = visualizer_dict.get("action_range", [-3,3])

        # History buffers
        self.pos_hist = deque([np.nan] * self.window_size, maxlen=self.window_size)
        self.reward_hist = deque([np.nan] * self.window_size, maxlen=self.window_size)
        self.avg_reward_hist = deque([np.nan] * self.window_size, maxlen=self.window_size)

        self.current_action_data = None
        self.sampled_action = None # To store the specific action taken

        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        gs = self.fig.add_gridspec(2, 2)

        # Top Plot: Position
        self.ax_pos = self.fig.add_subplot(gs[0, :])
        self.line_pos, = self.ax_pos.plot(list(self.pos_hist), color='blue', label='Position')
        self.ax_pos.set_xlim([0, self.window_size])
        self.ax_pos.set_ylim(self.pos_range)
        self.ax_pos.set_ylabel('Position')
        if self.goal_pos is not None:
            self.ax_pos.axhline(y=self.goal_pos, color='mediumvioletred', linestyle='--', label='Secret Goal Position')
        self.ax_pos.legend(loc='upper left')

        # Bottom Left: Reward
        self.ax_reward = self.fig.add_subplot(gs[1, 0])
        self.line_rewardAvg, = self.ax_reward.plot(list(self.avg_reward_hist), color='lightgreen', alpha=0.9, label='Avg Reward')
        self.line_reward, = self.ax_reward.plot(list(self.reward_hist), color='green', label='Current Reward')
        
        self.ax_reward.set_xlim([0, self.window_size])
        self.ax_reward.set_ylim(self.reward_range)
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.legend(loc='upper left')

        # Bottom Right: Action Distribution
        self.ax_probs = self.fig.add_subplot(gs[1, 1])
        if self.mode == 'discrete':
            num_actions = len(self.action_labels) if self.action_labels else 2
            self.bar_probs = self.ax_probs.bar(range(num_actions), [0]*num_actions, color='purple', alpha=0.6)
            self.ax_probs.set_ylim(0, 1.1)
            self.ax_probs.set_ylabel('Action Probability')
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
            self.ax_probs.set_xlabel('Position Adjustment')
            self.ax_probs.set_ylim(0, 1.0)
            self.ax_probs.set_ylabel('Probability Density')
            self.ax_probs.legend(loc='upper left')

        self.fig.tight_layout()

    def update_data(self, pos, reward, avg_reward, action_data, sampled_action):
        """
        Args:
            pos, reward, avg_reward: Scalar values
            action_data: 
                - If discrete: List of probabilities [p1, p2, ...]
                - If continuous: Tuple of (mean, std_dev)
            sampled_action:
                - If discrete: Integer index of the action taken
                - If continuous: Float value of the action taken
        """
        self.pos_hist.append(pos)
        self.reward_hist.append(reward)
        self.avg_reward_hist.append(avg_reward)
        self.current_action_data = action_data
        self.sampled_action = sampled_action

    def draw(self):
        if not self.is_open(): return

        # Update standard lines
        self.line_pos.set_ydata(list(self.pos_hist))
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

        if self.pos_range is None:
            self._smart_limit(self.ax_pos, [self.pos_hist, [self.goal_pos]] if self.goal_pos is not None else [self.pos_hist])
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