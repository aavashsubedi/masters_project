from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import wandb
import torch
import numpy as np
from scipy.special import kl_div
from math import dist
import matplotlib.pyplot as plt
from rl_utils import baseline_dists, MSE
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
        self.coordinates = torch.tensor(np.genfromtxt(coordinate_file, delimiter=',')).to(device)

        self.possible_agents = ["player_" + str(r) for r in range(self.agent_num)]
        # Mapping between agent name and ID
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: Discrete(self.num_nodes) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: MultiDiscrete([self.agent_num for _ in range(self.num_nodes)]) for agent in self.possible_agents
        } # This is the allocation space of all nodes, which can be allocated to any of n agents
        self.alloc = self._observation_spaces['player_0'].sample() # np.array([None for _ in range(self.num_nodes)]) # Allocated node list to use in baseline_dists calculation 

        self.render_mode = render_mode
        self.hists = None # Save histograms of baseline distances for rendering
        self.bin_centers = (torch.tensor([n/2 for n in range(8)][:-1], device=device) + \
                             torch.tensor([n/2 for n in range(8)][1:], device=device)) / 2

    def observation_space(self, agent):
        return MultiDiscrete([self.agent_num for _ in range(1, self.num_nodes)])

    def action_space(self, agent):
        return Discrete(self.graph.number_of_nodes())

    def observe(self, agent):
        return np.array(self.observations[agent])
    
    def calculate_rewards(self):
        for n in range(self.num_agents): # Update rewards
            array_sensitivity = sensitivity(len(np.where(self.alloc == n)[0]))
            array_resolution = resolution(1, baseline_dists(self.coordinates[np.where(self.alloc == n)]))
            #similarity = MSE(avg_hist, self.hists[n]) if not \
            #           np.isnan(MSE(avg_hist, self.hists[n])) else -10 # Similarity to an average
            self.rewards[self.agents[n]] = -MSE(array_sensitivity, self.target_sensitivity) - \
                                            MSE(array_resolution, self.target_resolution)
            
            #similarity # - #np.ma.masked_invalid(kl).sum()
        return array_sensitivity, array_resolution

    def close(self):
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the attributes
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: -10 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.alloc = self._observation_spaces['player_0'].sample()
        # Cyclic stepping through the agents list.
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

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
        # stores action of current agent
        self.state[self.agent_selection] = action
        self.alloc[action] = self.agent_name_mapping[agent]

        avg_reward = -10000 # Large negative will be overwritten
        for weighting_fn in (briggs_weighting, tapered_weighting):
            self.hists = [torch.histogram(baseline_dists(self.coordinates[np.where(self.alloc == i)]),
                                        bins=self.bin_centers,
                                        weights=weighting_fn(baseline_dists(self.coordinates[np.where(self.alloc == i)])),
                                    density=True)[0].to(device) for i in range(self.num_agents)]
            #kl = [kl_div(self.hists[i], self.hists[i+1]) for i in range(len(self.hists)-1)]
            #avg_hist = torch.average(self.hists, axis=0) # Used to calculate MSE

            sensitivity, resolution = self.calculate_rewards() # Updates self.rewards
            print(sensitivity, resolution)
            #wandb.log({"Sensitivity": sensitivity, "Resolution": resolution})

            if torch.mean(list(self.rewards.values())) > avg_reward:
                self.weighting_regime = weighting_fn

        wandb.log(self.rewards)
        # Update hists with best weighting regime
        self.hists = [torch.histogram(baseline_dists(self.coordinates[np.where(self.alloc == i)]),
                                    bins=self.bin_centers,
                                    weights=self.weighting_regime(baseline_dists(
                                        self.coordinates[np.where(self.alloc == i)])),
                                density=True)[0].to(device) for i in range(self.num_agents)]
            
        # The truncations dictionary must be updated for all players.
        #self.truncations = {
         #   agent: self.num_steps >= 200 for agent in self.agents # NOT USED
        #}

        # observe the current state
        for i in self.agents:
            self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]

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

        #if self.render_mode == "human":
         #   return self._render_frame(colours)
            
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
            fig, axes = plt.subplots(1, self.num_agents, figsize=(15, 5))
            for i in range(self.num_agents):
                axes[i].bar(self.bin_centers, self.hists[i])#, align='centre', width=0.5)

            fig.savefig(r'/share/nas2/lislaam/masters_project/src/rl_pipeline/SKA_histograms.png', bbox_inches='tight')
            wandb.log({"Histograms": wandb.Image(fig)})
            plt.close()
        
        return
    