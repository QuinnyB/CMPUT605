"""Temporal difference learning agent."""
import numpy as np
import math
from robotModuleFunctions import normalize, bin


class LearningAgent:
    """Temporal difference learning agent for gripper control."""
    
    def __init__(self, config):
        """
        Initialize the learning agent.
        
        Args:
            config: RobotConfig object containing learning parameters
        """
        self.config = config
        self.learning_cfg = config.learning
        self.gripper_cfg = config.gripper
        
        # Initialize weight vector
        feature_dim = self.learning_cfg.NUM_POS_BINS * self.learning_cfg.NUM_VEL_BINS
        self.w = np.zeros(feature_dim)
        
        # Initialize previous feature vector
        self.x_prev = np.zeros(feature_dim, dtype=int)
        
        # For verifier (retrospective validation)
        self.verifier_buffer_length = math.ceil(
            5 * (1 / (1 - self.learning_cfg.GAMMA))
        )
        self.verifier_indices = np.arange(self.verifier_buffer_length)
    
    def featurize(self, pos, vel):
        """
        Convert position and velocity to feature vector.
        
        Args:
            pos: Motor position (0-4095)
            vel: Motor velocity
            
        Returns:
            np.ndarray: One-hot encoded feature vector
        """
        # Normalize to [0, 1]
        pos_norm = normalize(pos, self.gripper_cfg.POSITION_CLOSED, 
                            self.gripper_cfg.POSITION_OPEN)
        vel_norm = normalize(vel, -self.gripper_cfg.MOTOR_VELOCITY, 
                            self.gripper_cfg.MOTOR_VELOCITY)
        
        # Bin the normalized values
        pos_bin = bin(pos_norm, self.learning_cfg.NUM_POS_BINS)
        vel_bin = bin(vel_norm, self.learning_cfg.NUM_VEL_BINS)
        
        # Create one-hot feature vector
        feature_dim = self.learning_cfg.NUM_POS_BINS * self.learning_cfg.NUM_VEL_BINS
        x = np.zeros(feature_dim, dtype=int)
        feature_idx = pos_bin * self.learning_cfg.NUM_VEL_BINS + vel_bin
        x[feature_idx] = 1
        
        return x
    
    def get_cumulant(self, load):
        """
        Convert load into cumulant signal (reward/penalty).
        
        Args:
            load: Motor load reading
            
        Returns:
            float: 1.0 if load exceeds threshold, 0.0 otherwise
        """
        return 1.0 if abs(load) > self.learning_cfg.LOAD_THRESHOLD else 0.0
    
    def compute_td_error(self, c, x):
        """
        Compute temporal difference error (delta).
        
        Args:
            c: Cumulant signal (current reward)
            x: Current feature vector
            
        Returns:
            float: TD error
        """
        gamma = self.learning_cfg.GAMMA
        delta = c + gamma * (self.w @ x) - (self.w @ self.x_prev)
        return delta
    
    def update_weights(self, delta, x_prev):
        """
        Update weight vector using TD learning rule.
        
        Args:
            delta: TD error
            x_prev: Previous feature vector
        """
        self.w = self.w + self.learning_cfg.ALPHA * delta * x_prev
    
    def predict(self, x):
        """
        Predict expected cumulative reward.
        
        Args:
            x: Feature vector
            
        Returns:
            float: Prediction
        """
        return self.w @ x * (1 - self.learning_cfg.GAMMA)
    
    def step(self, pos, vel, load):
        """
        Perform one learning step.
        
        Args:
            pos: Motor position
            vel: Motor velocity
            load: Motor load
            
        Returns:
            dict: Learning step data including features, cumulant, prediction, and TD error
        """
        # Extract features and cumulant
        x = self.featurize(pos, vel)
        c = self.get_cumulant(load)
        
        # Compute TD error
        delta = self.compute_td_error(c, x)
        
        # Update weights
        self.update_weights(delta, self.x_prev)
        
        # Make prediction
        pred = self.predict(x)
        
        # Update previous state
        self.x_prev = x
        
        return {
            'x': x,
            'c': c,
            'pred': pred,
            'delta': delta
        }
    
    def get_weights(self):
        """Return current weight vector."""
        return self.w.copy()
    
    def get_feature_vector(self):
        """Return current feature vector."""
        return self.x_prev.copy()
