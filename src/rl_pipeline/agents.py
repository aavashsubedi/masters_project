import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import wandb
from rl_utils import PolicyNetwork

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
        self.Q = torch.zeros((num_actions, num_actions), requires_grad=True) # Q-values for joint actions
        self.weights = (torch.ones(num_actions, num_actions, device=device) / num_actions).squeeze() # Stochastic policy
    
    def select_action(self, opponent_action):
        if opponent_action == None:
            opponent_action = 0
        # Convert Q-values to probabilities using softmax
        #probs = self.softmax(self.Q[:, opponent_action])
        policy = self.weights.unsqueeze(0)
        # Sample action from the probability distribution
        wandb.log({'{0} weights'.format(self.agent_id): self.weights})

        return torch.multinomial(policy[:, :, opponent_action], 1).item() # Without replacement

    def update_weights(self, own_action, opponent_action, own_reward, opponent_reward):
        # Update weights using policy gradient
        gradient = torch.zeros((self.num_actions, self.num_actions), device=device)
        gradient[own_action, opponent_action] = 1

        # Update Q-values using gradient descent
        q_values = self.Q[own_action, opponent_action]
        loss = -(own_reward + opponent_reward - q_values)
        loss.backward()
        
        with torch.no_grad():
            # Update Q-values
            self.Q.data -= self.learning_rate * self.Q.grad
            # Reset gradients
            self.Q.grad.zero_()
            # Update policy using softmax function
            self.weights = F.softmax(self.Q, dim=0)

        return None


class PPO:
    def __init__(self, num_actions, agent_id, learning_rate, exploration, 
                 gamma=0.99, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01): # Exploration not used
        self.policy_net = PolicyNetwork(num_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.num_actions = num_actions
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.agent_id = agent_id

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return dist, action.item()

    def update(self, rollout):
        states, actions, old_log_probs, rewards, next_states, dones = rollout
        returns = self._compute_returns(rewards)
        #returns = torch.FloatTensor(returns)
        #old_log_probs = torch.stack(old_log_probs)
        
        for _ in range(10):  # PPO epoch
            logits = self.policy_net(states)
            
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            wandb.log({'{0} weights'.format(self.agent_id): log_probs})
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * returns
            actor_loss = -torch.min(surr1, surr2).mean()
            
            values = self.policy_net(states) #.gather(1, actions.unsqueeze(1)).squeeze(1)
            value_loss = nn.MSELoss()(values, returns)

            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            import pdb; pdb.set_trace()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _compute_returns(self, rewards):
        returns = []
        R = 0
        #for r in reversed(rewards):
        R = rewards + self.gamma * R
        returns.insert(0, R)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # normalize returns
        return torch.tensor(rewards, device=device)
    