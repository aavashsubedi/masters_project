import math
import numpy as np
import torch

def baseline_dists(coordinates):
    dists = []

    for i in range(len(coordinates) - 1):
        for j in range(len(coordinates) - 1):
            x_i, y_i = coordinates[i][-2], coordinates[i][-1]
            x_j, y_j = coordinates[j][-2], coordinates[j][-1]
            euclidean_distance = (x_i - x_j)**2 + (y_i - y_j)**2
            euclidean_distance = euclidean_distance**0.5
            dists.append(euclidean_distance)

    return dists

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
    return np.square(np.subtract(experimental, simulated).mean())