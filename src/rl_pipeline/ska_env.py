import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from ska_ost_array_config.array_config import LowSubArray, MidSubArray
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from rl_utils import *
from astro_utils import*

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Heavily inspired by https://pettingzoo.farama.org/content/environment_creation/ tutorial
## CHANGED To gym env for single agent


class InterferometerEnv(gym.Env):
    """
    The "name" metadata allows the environment to be pretty printed.
    """
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, target_sensitivity, target_resolution,
                 num_nodes=197, num_arrays=2,
                 coordinate_file=r"/share/nas2/lislaam/masters_project/src/dataset/ska_xy_coordinates.csv", 
                 render_mode='human'):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_modes

        These attributes should not be changed after initialization.
        """
        self.num_nodes = num_nodes
        self.num_arrays = num_arrays
        self.target_sensitivity = target_sensitivity # Does nothing
        self.target_resolution = target_resolution # Does nothing yet
        self.weighting_regime = None
        self.coordinates = np.genfromtxt(coordinate_file, delimiter=',') # Needs to be CPU for plotting

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self.action_space = torch.eye(self.num_nodes)[torch.randperm(self.num_nodes)]  # Example action space
        self.observation_space = MultiDiscrete([self.num_arrays] * self.num_nodes)
        self.state = self.observation_space.sample() # Allocated node list to use in baseline_dists calculation 

        self.render_mode = render_mode
        self.hists = None # Save histograms of baseline distances for rendering
        self.bin_centers = (np.array([n/2 for n in range(8)][:-1]) + \
                             np.array([n/2 for n in range(8)][1:])) / 2

    def observation_space(self):
        return MultiDiscrete([self.num_arrays] * self.num_nodes) # (2,1,0,0,1,...)

    def action_space(self):
        return torch.eye(self.num_nodes)[torch.randperm(self.num_nodes)] # Set of permutations
        #return Discrete(self.graph.number_of_nodes()) # {0....196} # Change to permutation space
    
    def calculate_rewards(self):
        jensen_shannon = compute_jensen(self.hists[0], self.hists[1]) # Symmetric
        self.reward = - jensen_shannon 
        wandb.log({"J-S Divergence": jensen_shannon})
        return None

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the attributes
        """
        self.reward = 0
        self._cumulative_reward = 0
        self.termination = False
        self.truncation = False
        self.info = {}
        self.state = torch.tensor(self.observation_space.sample(), device=device).unsqueeze(0).unsqueeze(-1)
        self.observation = self.observation_space.sample()

        return self.state, self.info

    def step(self, action):
        """
        step(action) takes in an action and needs to update any internal state used by observe() or render()
        """
        #import pdb; pdb.set_trace()
        self.state = torch.matmul(action, self.state.float()).long().to(device) # Update state by multiplying by permutation matrix
        self.hists = [np.histogram(baseline_dists(self.coordinates[torch.where(self.state == i)[0].tolist()]),
                                    bins=np.array([n/2 for n in range(8)]),
                                    density=True)[0] for i in range(self.num_arrays)]

        self.hists = [self.hists[i] / np.sum(self.hists[i]) for i in range(len(self.hists))] # Normalise histograms

        self.calculate_rewards() # Also updates self.reward
        self.observation = self.state # observe the current state (MDP not POMDP)

        if self.render_mode == "human":
            self.render()

        return self.observation, self.reward, self.termination, self.truncation


    def render(self):
        colours = []
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.num_arrays)))
        for i in range(self.num_arrays):
            colours.append(next(color))
            
        for i in range(self.num_arrays):
            for n in range(self.num_nodes):
                if self.state[0][n] == i: # 0 is the batch dimension
                    plt.plot(self.coordinates[n, 0], self.coordinates[n, 1], '.', color=colours[i], 
                                label='Array {}'.format(i+1))
                    
        #plt.legend()
        plt.xlim(-40, 60)
        plt.ylim(-70, 70)
        plt.savefig(r'/share/nas2/lislaam/masters_project/src/rl_pipeline/SKA_allocation.png', bbox_inches='tight')
        wandb.log({"Allocation": wandb.Image(plt)}) 
        plt.close()

        if self.hists != None:
            if len(self.hists) <= 5:
                fig, axes = plt.subplots(1, self.num_arrays, figsize=(15, 5))
                if self.num_arrays == 1:
                    axes.bar(self.bin_centers, self.hists[0], align='center', width=0.5)
                else:
                    for i in range(self.num_arrays):
                        axes[i].bar(self.bin_centers, self.hists[i], align='center', width=0.5)
                        axes[i].set_ylim(0, 0.6)
            
            else:
                fig, axes = plt.subplots(2, int((self.num_arrays+1)/2), figsize=(15, 5))

                for i in range(int((self.num_arrays+1)/2)):
                    axes[0][i].bar(self.bin_centers, self.hists[i], align='center', width=0.5)
                    axes[0][i].set_ylim(0, 1.1)
                    try:
                        axes[1][i].bar(self.bin_centers, self.hists[int((self.num_arrays+1)/2)+i],
                                        align='center', width=0.5)
                        axes[1][i].set_ylim(0, 1.1)
                    except IndexError:
                        pass
        
            fig.savefig(r'/share/nas2/lislaam/masters_project/src/rl_pipeline/SKA_histograms.png', bbox_inches='tight')
            wandb.log({"Histograms": wandb.Image(fig)})
            plt.close()
        
        return
    