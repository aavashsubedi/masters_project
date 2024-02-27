import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
#from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SGD:
    def __init__(self, num_actions, agent_id, learning_rate):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.weights = torch.ones(num_actions, device=device) * 0.2 # Optimistic initialisation
    
    def select_action(self, state):
        # Select action using epsilon-greedy policy
        if np.random.rand() < 0.1: # Exploration percentage
            wandb.log({'{0} weights'.format(self.agent_id): self.weights})
            return np.random.choice(self.num_actions)
        else:
            wandb.log({'{0} weights'.format(self.agent_id): self.weights})
            return torch.argmax(self.weights)

    def update_weights(self, state, action, reward):
        # Update weights using stochastic gradient descent
        self.weights[action] += self.learning_rate * reward[self.agent_id]


class LOLA:
    def __init__(self, num_actions, agent_id, learning_rate):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.weights = torch.ones(num_actions, device=device) * 0.2 # Optimistic initialisation
    
    def select_action(self, state):
        # Select action using LOLA (Foerster 2018)
        if np.random.rand() < 0.1: # Exploration percentage
            wandb.log({'{0} weights'.format(self.agent_id): self.weights})
            return np.random.choice(self.num_actions)
        else:
            wandb.log({'{0} weights'.format(self.agent_id): self.weights})
            return torch.argmax(self.weights)

    def update_weights(self, state, action, reward):
        # Update weights using stochastic gradient descent
        self.weights[action] += self.learning_rate * reward[self.agent_id]




class PPO(nn.Module):
    def __init__(self, num_actions):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            self._layer_init(nn.Linear(128 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    