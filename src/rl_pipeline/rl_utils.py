import numpy as np
import math
import torch
import wandb
import omegaconf
from scipy.spatial.distance import jensenshannon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baseline_dists(coordinates):
    antennas = len(coordinates) #.size()
    return np.array([math.dist(x,y) for x in coordinates for y in coordinates])#, device=device)


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


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'project': cfg.project_name, 'config': config_dict, 'reinit': True, 'mode': cfg.wandb,
              'settings': wandb.Settings(_disable_stats=True)}
    run = wandb.init(**kwargs)
    #wandb.save('*.txt')
    #run.save()
    return cfg, run


def compute_jensen(hist_1, hist_2):
    #convert the histogram to a probability distribution
    #first compute the histogram
    #hist_1, bin_edges_1 = np.histogram(value_1, bins=5, density=True)
    #hist_2, bin_edges_2 = np.histogram(value_2, bins=5, density=True)
    #compute the jensen shannon divergence
    hist_1 = hist_1 / np.sum(hist_1)
    hist_2 = hist_2 / np.sum(hist_2)
    return jensenshannon(hist_1, hist_2)