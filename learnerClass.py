'''
Actor-critic Learner class for CMPUT 605 Robot Module 4
Written by: Quinn Boser, with assistance from Google Gemini 
March 2026
'''

import math
import numpy as np
from helperFunctions import compute_softmax_prob, get_softmax_action

class ACLearner_Discrete:
    def __init__(self, agent_dict = {}):
        """Setup for the agent called when the experiment first starts.
        Assume agent_dict contains:
        {
            "actor_alpha": float,
            "critic_alpha": float,
            "avg_reward_alpha": float,
            "num_actions": int,
            "feature_vector_length": int,
            "initial_actor_w": float or None (default 0.0),
            "initial_critic_w": float or None (default 0.0),
            "initial_avg_reward": float or None (default 0.0),
        }
        """
        # Set parameters from agent_dict
        self.actor_alpha = agent_dict.get("actor_alpha")
        self.critic_alpha = agent_dict.get("critic_alpha")
        self.avg_reward_alpha = agent_dict.get("avg_reward_alpha")
        self.num_actions = agent_dict.get("num_actions")
        feature_vector_length = agent_dict.get("feature_vector_length")
        initial_actor_w = agent_dict.get("initial_actor_w", 0.0)
        initial_critic_w = agent_dict.get("initial_critic_w", 0.0)
        self.avg_reward = agent_dict.get("initial_avg_reward", 0.0)

        # Initialize things
        self.x_cur = np.zeros(feature_vector_length, dtype=int) 
        self.actor_w = np.full((feature_vector_length, self.num_actions), initial_actor_w)
        self.critic_w = np.full(feature_vector_length, initial_critic_w)
        self.last_action = None 
        self.softmax_probs = compute_softmax_prob(self.actor_w, self.x_cur)
    
    # Update weights 
    def update(self, reward_next, x_next, learning_enabled = True):
        if not learning_enabled:
            self.x_cur = x_next
            return
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
    
class ACLearner_Continuous:
    def __init__(self, agent_dict = {}):
        """Setup for the agent called when the experiment first starts.
        Assume agent_dict contains:
        {
            "feature_vector_length": int,
            "avg_reward_alpha": float,
            "critic_alpha": float,
            "actor_alpha": float,
            "lambda_actor": float or None (default None, meaning no eligibility traces for actor),
            "lambda_critic": float or None (default None, meaning no eligibility traces for critic),
            "initial_actor_mu": float or None (default 0.0),
            "initial_actor_sd": float or None (default 0.0),
            "initial_critic_w": float or None (default 0.0),
            "initial_avg_reward": float or None (default 0.0),
        }
        """
        # Set parameters from agent_dict
        feature_vector_length = agent_dict.get("feature_vector_length")
        self.avg_reward_alpha = agent_dict.get("avg_reward_alpha")
        self.critic_alpha = agent_dict.get("critic_alpha")
        self.actor_alpha_mu = agent_dict.get("actor_alpha_mu")
        self.actor_alpha_sd = agent_dict.get("actor_alpha_sd")
        self.lambda_actor = agent_dict.get("lambda_actor", 0.0)
        self.lambda_critic = agent_dict.get("lambda_critic", 0.0)
        initial_actor_mu = agent_dict.get("initial_actor_w", 0.0)
        initial_actor_sd = agent_dict.get("initial_actor_w", 0.0)
        initial_critic_w = agent_dict.get("initial_critic_w", 0.0)
        self.avg_reward = agent_dict.get("initial_avg_reward", 0.0)

        # Initialize things
        self.x_cur = np.zeros(feature_vector_length, dtype=int) # Current state feature vector
        self.critic_e = np.zeros(feature_vector_length)         # For eligibility trace vector for critic
        self.actor_e_mu = np.zeros(feature_vector_length)       # For eligibility trace vector for actor mean
        self.actor_e_sd = np.zeros(feature_vector_length)       # For eligibility trace vector for actor standard deviation
        self.critic_w = np.full(feature_vector_length, initial_critic_w) # Critic weight vector
        self.actor_w_mu = np.full((feature_vector_length), initial_actor_mu) # Actor weight vector for mean
        self.actor_w_sd = np.full((feature_vector_length), initial_actor_sd) # Actor weight vector for standard deviation
        self.last_action = None 
        self.mu = 0.0
        self.sd = 1.0
    
    # Update weights 
    def update(self, reward_next, x_next):
        # print(f"Before update: avg_reward={self.avg_reward}, critic_w={self.critic_w}, actor_w={self.actor_w}")
        # Calculate TD error: delta = reward_next - avg_reward + critic_w*x_next - critic_w*x_cur
        delta = reward_next - self.avg_reward + (self.critic_w @ x_next) - (self.critic_w @ self.x_cur)
        # Update average reward: avg_reward = avg_reward + avg_reward_alpha * delta
        self.avg_reward += self.avg_reward_alpha * delta
        # Update critic eligibility traces: 
        self.critic_e = self.lambda_critic * self.critic_e + self.x_cur
        # Update actor eligibility traces:
        self.actor_e_mu = self.lambda_actor * self.actor_e_mu + (self.last_action - self.mu) * self.x_cur
        self.actor_e_sd = self.lambda_actor * self.actor_e_sd + ((self.last_action - self.mu)**2 - self.sd**2) * self.x_cur
        # Update critic weights: 
        self.critic_w += self.critic_alpha * delta * self.critic_e
        # Update actor weights:
        print(f"Updating actor weights with delta={delta}, last_action={self.last_action}, mu={self.mu}, sd={self.sd}")
        self.actor_w_mu += self.actor_alpha_mu * delta * self.actor_e_mu
        self.actor_w_sd += self.actor_alpha_sd * delta * self.actor_e_sd
        # Update x_cur 
        self.x_cur = x_next
        return
    
    # Get action to take in current state
    def get_next_action(self):
        # Calculate mean and standard deviation for current state
        self.mu = self.actor_w_mu @ self.x_cur
        self.sd = np.exp(np.clip(self.actor_w_sd @ self.x_cur,-20,2)) # clip ln of sd to avoid overflow
        # Select action from normal distribution with mean mu and standard deviation sd
        action = np.random.normal(self.mu, self.sd)
        # Store last action taken (for use in update step)
        self.last_action = action
        print(f"Selected action {action} with mean {self.mu} and std dev {self.sd}")
        return action

 
