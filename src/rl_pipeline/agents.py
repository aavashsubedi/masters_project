import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import wandb
#from supersuit import color_reduction_v0, frame_stack_v1, resize_v1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""All agents need a select_action() and update() function"""

class SGD:
    def __init__(self, num_actions, agent_id, learning_rate, exploration=0.1):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.weights = torch.ones(num_actions, device=device) * 0 # Optimistic initialisation
    
    def select_action(self, state):
        # Select action using epsilon-greedy policy
        if np.random.rand() < self.exploration: # Exploration percentage
            wandb.log({'{0} weights'.format(self.agent_id): self.weights})
            return np.random.choice(self.num_actions)
        else:
            wandb.log({'{0} weights'.format(self.agent_id): self.weights})
            return torch.argmax(self.weights)

    def update_weights(self, state, action, reward):
        # Update weights using stochastic gradient descent
        self.weights[action] += self.learning_rate * reward[self.agent_id]


class LOLA(nn.Module):
    def __init__(self, num_actions, agent_id, learning_rate, exploration): # exploration not used.
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.Q = nn.Parameter(torch.zeros((num_actions, num_actions)), requires_grad=True) # Q-values for joint actions
        self.softmax = nn.Softmax(dim=1)
        self.weights = (torch.ones(num_actions, num_actions, device=device) / num_actions).squeeze() # Stochastic policy
    
    def select_action(self, opponent_action):
        if opponent_action == None:
            opponent_action = 0
        # Convert Q-values to probabilities using softmax
        probs = self.softmax(self.Q[:, opponent_action])
        # Sample action from the probability distribution
        wandb.log({'{0} weights'.format(self.agent_id): self.weights})
        return torch.multinomial(probs, 1, replacement=True).item() # torch.multinomial(self.weights[:, opponent_action].squeeze(), num_samples=1, replacement=True).item()

    def update_weights(self, own_action, opponent_action, own_reward, opponent_reward):
        # Update weights using policy gradient
        gradient = torch.zeros((self.num_actions, self.num_actions), device=device)
        gradient[own_action, opponent_action] = 1

        # Update Q-values using gradient descent
        self.Q += self.learning_rate * (own_reward + opponent_reward - self.Q[own_action, opponent_action]) * gradient
        # Update policy using softmax function
        self.weights = torch.softmax(self.Q, dim=0)

        return None


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
    