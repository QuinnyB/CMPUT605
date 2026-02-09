"""Real-time visualization of robot learning and motion data."""
import matplotlib.pyplot as plt
from collections import deque
import numpy as np


class Visualizer:
    """Real-time plotting of motor state and learning metrics."""
    
    def __init__(self, config):
        """
        Initialize the visualizer with plots and history buffers.
        
        Args:
            config: RobotConfig object containing visualization settings
        """
        self.config = config
        self.viz_cfg = config.visualization
        self.gripper_cfg = config.gripper
        
        # Initialize history buffers
        self.pos_history = deque([self.gripper_cfg.POSITION_CLOSED] * self.viz_cfg.WINDOW_SIZE, 
                                maxlen=self.viz_cfg.WINDOW_SIZE)
        self.vel_history = deque([0] * self.viz_cfg.WINDOW_SIZE, 
                                maxlen=self.viz_cfg.WINDOW_SIZE)
        self.load_history = deque([0] * self.viz_cfg.WINDOW_SIZE, 
                                 maxlen=self.viz_cfg.WINDOW_SIZE)
        self.cumulant_history = deque([0] * self.viz_cfg.WINDOW_SIZE, 
                                     maxlen=self.viz_cfg.WINDOW_SIZE)
        self.pred_history = deque([0] * self.viz_cfg.WINDOW_SIZE, 
                                 maxlen=self.viz_cfg.WINDOW_SIZE)
        self.verifier_history = deque([np.nan] * self.viz_cfg.WINDOW_SIZE, 
                                     maxlen=self.viz_cfg.WINDOW_SIZE)
        
        # Create plots
        self._setup_plots()
    
    def _setup_plots(self):
        """Set up matplotlib figure and axes."""
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=self.viz_cfg.FIGSIZE)
        self.fig.suptitle(f"Motor {self.gripper_cfg.MOTOR_ID}: Position, Velocity, and Load")
        
        # --- Subplot 1: Position and Velocity ---
        self.axs[0].set_xlabel('Time (Samples)')
        self.axs[0].set_ylabel('Motor Position (0-4095)', color='blue')
        self.line_pos, = self.axs[0].plot(list(self.pos_history), color='blue', label='Position')
        self.axs[0].tick_params(axis='y', labelcolor='blue')
        self.axs[0].set_ylim(0, 4095)
        self.axs[0].grid(True, alpha=0.3)
        
        # Right axis for velocity
        self.axs0_vel = self.axs[0].twinx()
        self.axs0_vel.set_ylabel('Velocity', color='green')
        self.line_vel, = self.axs0_vel.plot(list(self.vel_history), color='green', 
                                           alpha=0.6, label='Velocity')
        self.axs0_vel.tick_params(axis='y', labelcolor='green')
        self.axs0_vel.set_ylim(-100, 100)
        
        # --- Subplot 2: Load ---
        self.axs[1].set_xlabel('Time (Samples)')
        self.axs[1].set_ylabel('Raw Load (0-2000)', color='red')
        self.line_load, = self.axs[1].plot(list(self.load_history), color='red', 
                                          alpha=0.6, label='Load')
        self.axs[1].tick_params(axis='y', labelcolor='red')
        self.axs[1].set_ylim(-400, 400)
        self.axs[1].grid(True, alpha=0.3)
        
        # --- Subplot 3: Cumulant and Prediction ---
        self.axs[2].set_xlabel('Time (Samples)')
        self.axs[2].set_ylabel('Cumulant Signal / Prediction', color='purple')
        self.line_c, = self.axs[2].plot(list(self.cumulant_history), color='orange', 
                                       alpha=0.6, label='Cumulant Signal')
        self.line_pred, = self.axs[2].plot(list(self.pred_history), color='purple', 
                                          alpha=0.6, label='Prediction')
        self.line_verifier, = self.axs[2].plot(list(self.verifier_history), color='brown', 
                                              alpha=0.6, label='Verifier')
        self.axs[2].tick_params(axis='y', labelcolor='purple')
        self.axs[2].set_ylim(-0.5, 1.5)
        self.axs[2].grid(True, alpha=0.3)
        
        self.fig.tight_layout()
    
    def update(self, pos, vel, load, cumulant, prediction, verifier_value=None):
        """
        Update all history buffers and refresh plots.
        
        Args:
            pos: Motor position
            vel: Motor velocity
            load: Motor load
            cumulant: Cumulant signal value
            prediction: TD prediction value
            verifier_value: (Optional) Verifier/retrospective TD value
        """
        # Append to histories
        self.pos_history.append(pos)
        self.vel_history.append(vel)
        self.load_history.append(load)
        self.cumulant_history.append(cumulant)
        self.pred_history.append(prediction)
        
        if verifier_value is not None:
            self.verifier_history.append(verifier_value)
        else:
            self.verifier_history.append(np.nan)
        
        # Update line data
        self.line_pos.set_ydata(list(self.pos_history))
        self.line_vel.set_ydata(list(self.vel_history))
        self.line_load.set_ydata(list(self.load_history))
        self.line_c.set_ydata(list(self.cumulant_history))
        self.line_pred.set_ydata(list(self.pred_history))
        self.line_verifier.set_ydata(list(self.verifier_history))
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def flush_events(self):
        """Flush matplotlib events (for window responsiveness during pause)."""
        self.fig.canvas.flush_events()
    
    def is_open(self):
        """Check if the figure window is still open."""
        return plt.fignum_exists(self.fig.number)
    
    def close(self):
        """Close the visualization window."""
        plt.close(self.fig)
    
    def get_history(self):
        """
        Get all current history buffers.
        
        Returns:
            dict: Dictionary containing all history deques
        """
        return {
            'pos': self.pos_history,
            'vel': self.vel_history,
            'load': self.load_history,
            'cumulant': self.cumulant_history,
            'prediction': self.pred_history,
            'verifier': self.verifier_history
        }
