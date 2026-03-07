'''
TD Learner class for CMPUT 605 Robot Module 4
Written by: Quinn Boser, with assistance from Google Gemini 
March 2026
'''

import math
import numpy as np
from RM4_helperFunctions import compute_softmax_prob, get_softmax_action

class ACLearner:
    def __init__(self, agent_dict = {}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_dict contains:
        {
            "actor_alpha": float,
            "critic_alpha": float,
            "avg_reward_alpha": float,
            "num_actions": int,
            "feature_vector_length": int,
            "initial_actor_w": float or None,
            "initial_critic_w": float or None,
        }
        """
        # Set parameters from agent_dict
        self.actor_alpha = agent_dict.get("actor_alpha")
        self.critic_alpha = agent_dict.get("critic_alpha")
        self.avg_reward_alpha = agent_dict.get("avg_reward_alpha")
        self.num_actions = agent_dict.get("num_actions")

        # Initialize things
        feature_vector_length = agent_dict.get("feature_vector_length")
        initial_actor_w = agent_dict.get("initial_actor_w", 0.0)
        initial_critic_w = agent_dict.get("initial_critic_w", 0.0)
        self.x_cur = np.zeros(feature_vector_length, dtype=int) 
        self.actor_w = np.full((feature_vector_length, self.num_actions), initial_actor_w)
        self.critic_w = np.full(feature_vector_length, initial_critic_w)
        self.avg_reward = 0.0
        self.last_action = None 
        self.softmax_probs = compute_softmax_prob(self.actor_w, self.x_cur)
    
    # Update weights 
    def update(self, reward_next, x_next):
        # print(f"Before update: avg_reward={self.avg_reward}, critic_w={self.critic_w}, actor_w={self.actor_w}")
        # Calculate TD error: delta = reward_next - avg_reward + critic_w*x_next - critic_w*x_cur
        delta = reward_next - self.avg_reward + (self.critic_w @ x_next) - (self.critic_w @ self.x_cur)
        # Update average reward: avg_reward = avg_reward + avg_reward_alpha * delta
        self.avg_reward += self.avg_reward_alpha * delta
        # Update critic weights: critic_w = critic_w + critic_alpha * delta * x_cur
        self.critic_w += self.critic_alpha * delta * self.x_cur
        # Update actor weights: actor_w = actor_w + actor_alpha * delta * (x_h - softmax_probs))
        x_h = x_h = np.eye(self.num_actions)[self.last_action]  # First construct x_h(s,a) - all zereos except for index corresponding to last action taken
        self.actor_w += self.actor_alpha * delta * (x_h - self.softmax_probs) * self.x_cur[:, np.newaxis]  # Use broadcasting to update only the column corresponding to last state
        # Update x_cur 
        self.x_cur = x_next
        return
    
    # Get action to take in current state
    def get_next_action(self):
        # Calculate softmax probabilities for current state: softmax_probs = compute_softmax_prob(actor_w, x_cur)
        self.softmax_probs = compute_softmax_prob(self.actor_w, self.x_cur)
        # Select action based on softmax probabilities: action = get_softmax_action(softmax_probs)
        action = get_softmax_action(self.softmax_probs)
        # Store last action taken (for use in update step)
        self.last_action = action
        print(f"Selected action {action} with softmax probabilities {self.softmax_probs}")
        return action

 
