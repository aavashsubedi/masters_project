import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

class SGD_Agent:
    def __init__(self, num_actions, agent_id, learning_rate=0.1):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.weights = np.zeros(num_actions)
    
    def select_action(self, state):
        # Select action using epsilon-greedy policy
        if np.random.rand() < 0.3: # Randomness level 10 percent
            return np.random.choice(self.num_actions)
        else:
            print(self.weights)
            return np.argmax(self.weights)

    def update_weights(self, state, action, reward):
        # Update weights using stochastic gradient descent
        self.weights[action] += self.learning_rate * reward[self.agent_id]


class PPOAgent(nn.Module):
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
    