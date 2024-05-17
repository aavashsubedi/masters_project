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

def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = InterferometerEnv(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class InterferometerEnv(AECEnv):
    """
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, target_sensitivity, target_resolution,
                 num_nodes=197, num_agents=2,
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
        self.agent_num = num_agents
        self.target_sensitivity = target_sensitivity
        self.target_resolution = target_resolution
        self.weighting_regime = None
        self.coords = np.genfromtxt(coordinate_file, delimiter=',')
        self.coordinates = self.coords[np.random.choice(self.coords.shape[0], num_nodes, replace=False), :] # self.coords[0:self.num_nodes,:] # Needs to be CPU for plotting

        self.possible_agents = ["player_" + str(r) for r in range(self.agent_num)]
        # Mapping between agent name and ID
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: Discrete(self.num_nodes) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: MultiDiscrete([self.agent_num for _ in range(self.num_nodes)]) for agent in self.possible_agents
        } # This is the allocation space of all nodes, which can be allocated to any of n agents
        self.alloc = self._observation_spaces['player_0'].sample() # Allocated node list to use in baseline_dists calculation 

        self.render_mode = render_mode
        self.hists = None # Save histograms of baseline distances for rendering
        self.bin_centers = (np.array([n/2 for n in range(8)][:-1]) + \
                             np.array([n/2 for n in range(8)][1:])) / 2

    def observation_space(self, agent):
        return MultiDiscrete([self.agent_num for _ in range(1, self.num_nodes)]) # {(0,1,0,0,1,)}

    def action_space(self, agent):
        return Discrete(self.graph.number_of_nodes()) # {0....196} # Change to permutation

    def observe(self, agent):
        return np.array(self.observations[agent])

    def calculate_rewards(self):
        avg_hist = np.mean(self.hists, axis=0)
        self.rewards = {agent: 0 for agent in self.agents}
        for n in range(self.num_agents): # Update rewards
            jensen_shannon = compute_jensen(self.hists[n], avg_hist) # Symmetric
            self.rewards[self.agents[n]] -= jensen_shannon
        #wandb.log({"J-S Divergence": jensen_shannon})
        return #array_sensitivity, array_resolution

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the attributes
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: torch.tensor(self._observation_spaces['player_0'].sample(), device=device) for agent in self.agents}
        self.observations = {agent: self._observation_spaces[agent].sample() for agent in self.agents}
        self.alloc = torch.tensor(self._observation_spaces['player_0'].sample(), device=device)
        # Cyclic stepping through the agents list.
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.coordinates = self.coords[np.random.choice(self.coords.shape[0], self.num_nodes, replace=False), :] # New coords

        return self.observations, self.infos

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update any internal state used by observe() or render()
        """
        if (self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]):
            # handles stepping an agent which is already finished
            return

        agent = self.agent_selection
        # Label each node for which agent it is allocated to
        self.alloc[action] = self.agent_name_mapping[agent]
        self.state[self.agent_selection] = self.alloc
        self.hists = [np.histogram(baseline_dists(self.coordinates[torch.where(self.alloc == i)[0].tolist()]),
                                    bins=np.array([n/2 for n in range(8)]), #bins=8,# min=0, max=4,
                                    #weights=weighting_fn(baseline_dists(self.coordinates[np.where(self.alloc == i)])), 
                                    density=True)[0] for i in range(self.num_agents)]

        self.hists = [self.hists[i] / np.sum(self.hists[i]) for i in range(len(self.hists))] # Normalise histograms

        self.calculate_rewards() # Updates self.rewards
        #wandb.log(self.rewards)

        # observe the current state
        for i in self.agents:
            self.observations[i] = self.state[self.agents[self.agent_name_mapping[i]]]

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        return self.observations, self.rewards, self.terminations, self.truncations


    def render(self):
        colours = []
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.num_agents)))
        for i in range(self.num_agents):
            colours.append(next(color))
            
        for i in range(self.num_agents):
            for n in range(self.num_nodes):
                if self.alloc[n] == i:
                    plt.plot(self.coordinates[n, 0], self.coordinates[n, 1], '.', color=colours[i], 
                                label='Agent {}'.format(i+1))
                    
        #plt.legend()
        plt.xlim(-40, 60)
        plt.ylim(-70, 70)
        plt.savefig(r'/share/nas2/lislaam/masters_project/src/rl_pipeline/SKA_allocation.png', bbox_inches='tight')
        wandb.log({"Allocation": wandb.Image(plt)}) 
        plt.close()

        if self.hists != None:
            if len(self.hists) <= 5:
                fig, axes = plt.subplots(1, self.num_agents, figsize=(15, 5))
                if self.num_agents == 1:
                    axes.bar(self.bin_centers, self.hists[0], align='center', width=0.5)
                else:
                    for i in range(self.num_agents):
                        axes[i].bar(self.bin_centers, self.hists[i], align='center', width=0.5)
                        axes[i].set_ylim(0, 0.6)
            
            else:
                fig, axes = plt.subplots(2, int((self.num_agents+1)/2), figsize=(15, 5))

                for i in range(int((self.num_agents+1)/2)):
                    axes[0][i].bar(self.bin_centers, self.hists[i], align='center', width=0.5)
                    axes[0][i].set_ylim(0, 0.6)
                    try:
                        axes[1][i].bar(self.bin_centers, self.hists[int((self.num_agents+1)/2)+i],
                                        align='center', width=0.5)
                        axes[1][i].set_ylim(0, 0.6)
                    except IndexError:
                        pass
        
            # fig.savefig(r'/share/nas2/lislaam/masters_project/src/rl_pipeline/SKA_histograms.png', bbox_inches='tight')
            wandb.log({"Histograms": wandb.Image(fig)})
            plt.close()
        
        return
    