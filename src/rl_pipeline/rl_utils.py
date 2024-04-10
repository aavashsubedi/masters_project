import numpy as np
import random
import torch
import torch.nn as nn
import wandb
import omegaconf
from scipy.spatial.distance import jensenshannon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# https://github.com/openai/baselines/blob/master/baselines/ddpg/memory.py

class RingBuffer:
    def __init__(self, maxlen, shape, dtype='torch.FloatTensor'):
        self.maxlen = maxlen
        self.start = 0 # the idx of the 0th element in the buffer
        self.length = 0
        self.data = torch.zeros(maxlen, *shape, device=device).type(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        torch_idxs = torch.from_numpy((self.start + idxs) % self.maxlen, device=device).long()
        return self.data[torch_idxs]


    def append(self, v):
        batch_size = v.shape[0]
        for i in range(batch_size):
            if self.length < self.maxlen:
                # We have space, simply increase the length.
                self.length += 1
            elif self.length == self.maxlen:
                # No space, "remove" the first item.
                self.start = (self.start + 1) % self.maxlen
            else:
                # This should never happen.
                raise RuntimeError()
            self.data[(self.start + self.length - 1) % self.maxlen] = v[i]


class Memory:
    def __init__(self, limit, action_shape, observation_shape):
        self.limit = limit

        self.observations = RingBuffer(limit, observation_shape)
        self.discrete_actions = RingBuffer(limit, action_shape, dtype='torch.ByteTensor')
        self.dense_actions = RingBuffer(limit, action_shape)
        self.rewards = RingBuffer(limit, [1])

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.random_integers(self.nb_entries - 2, size=batch_size)

        obs_batch = self.observations.get_batch(batch_idxs)
        discrete_actions_batch = self.discrete_actions.get_batch(batch_idxs)
        dense_actions_batch = self.dense_actions.get_batch(batch_idxs)
        reward_batch = self.rewards.get_batch(batch_idxs)

        return obs_batch, discrete_actions_batch, dense_actions_batch, reward_batch

    def append(self, obs, discrete_action, dense_action, reward):
        self.observations.append(obs)
        self.discrete_actions.append(discrete_action)
        self.dense_actions.append(dense_action)
        self.rewards.append(reward)

    @property
    def nb_entries(self):
        return len(self.observations)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Output range: [-1, 1]
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)
        return (
            torch.stack(states).to(device),
            actions, 
            rewards,
            torch.stack(next_states).to(device),
            dones
        )

    def __len__(self):
        return len(self.buffer)


class Transition:
    def __init__(self, *args):
        self.state = []
        self.action = []
        self.next_state = []
        self.reward = []
        self.done = []
        for transition in args:
            self.state.append(transition[0])
            self.action.append(transition[1])
            self.reward.append(transition[2])
            self.next_state.append(transition[3])
            self.done.append(transition[4])


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_actions, 64)
        self.fc_pi = nn.Linear(64, num_actions)

    def forward(self, x):
        #x = x.to(torch.float32)
        x = self.relu(self.fc(x))
        #import pdb; pdb.set_trace()
        logits = self.fc_pi(x)
        return logits


def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0).astype(np.float32)
    # transpose to be (batch, channel, height, width)
    #obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.from_numpy(obs).to(device)

    return obs

def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x

def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}

    return x

def MSE(experimental, simulated):
    return torch.square(torch.subtract(float(experimental), float(simulated)).mean())

def compute_jensen(hist_1, hist_2):
    # Normalise histograms
    hist_1 = hist_1 / np.sum(hist_1)
    hist_2 = hist_2 / np.sum(hist_2)
    return jensenshannon(hist_1, hist_2)

# The reward of players 1 or 2 (2 player case for now)
def _grad_reward(reward_func): return torch.autograd.grad(reward_func)

def grad_rewards(weight, agents=[0,1]):
    return torch.concatenate([_grad_reward(weight)[agents[0]], _grad_reward(weight)[agents[1]]])

def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'project': cfg.project_name, 'config': config_dict, 'reinit': True, 'mode': cfg.wandb,
              'settings': wandb.Settings(_disable_stats=True)}
    run = wandb.init(**kwargs)
    
    return cfg, run
