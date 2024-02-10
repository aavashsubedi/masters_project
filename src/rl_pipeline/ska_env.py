import gym
from gym import spaces
from gymnasium.spaces import Dict, Box, Discrete, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import functools

import networkx as nx
import numpy as np
from scipy.special import kl_div
import itertools
from math import dist
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rl_utils import baseline_dists

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

    def __init__(self, num_steps=100, num_nodes=197, num_agents=2, coordinate_file=r"src\dataset\ska_xy_coordinates.csv", render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_modes

        These attributes should not be changed after initialization.
        """
        self.num_nodes = num_nodes
        self.num_steps = num_steps

        # Creating graph. Keeping this for the sake of plotting
        self.coordinates = np.genfromtxt(coordinate_file, delimiter=',')
        self.graph = nx.Graph()
        self.graph.add_nodes_from([(i, {'coordinates': self.coordinates[i], 
                                        'cluster': None}) for i in range(len(self.coordinates))])
        self.graph.add_edges_from(itertools.combinations(self.graph.nodes, 2))

        self.possible_agents = ["player_" + str(r) for r in range(num_agents)]

        # Mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: Discrete(self.graph.number_of_nodes()) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: MultiDiscrete([self.num_nodes for n in range(self.num_nodes+1)]) for agent in self.possible_agents
        } # This is the allocation of all nodes, which can be allocated to any of n agents
        self.alloc = np.array([None for _ in range(self.num_nodes)]) # Allocated node list to use in baseline_dists calculation 
        self.render_mode = render_mode
        self.hists = None # Save histograms of baseline distances for rendering
        self.bin_centers = (np.array([n/2 for n in range(8)][:-1]) + np.array([n/2 for n in range(8)][1:])) / 2

    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete([n for n in range(1, self.num_nodes+1)])

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.graph.number_of_nodes())

    def observe(self, agent):
        return np.array(self.observations[agent])

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
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        return self.observations, self.infos

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            #self._was_dead_step(action)
            return

        agent = self.agent_selection
        # stores action of current agent
        self.state[self.agent_selection] = action
        self.alloc[action] = self.agent_name_mapping[agent]

        # rewards for all agents are placed in the .rewards dictionary
        # Could introduce baseline weights here!!!! Start with uniform for now
        if self.num_steps < 10: # Can't histogram one move..
            self.rewards[self.agent_selection] = -10

        else:
            self.hists = [np.histogram(baseline_dists(self.coordinates[np.where(self.alloc == i)]), bins=[n/2 for n in range(8)], 
                                    density=True)[0] for i in range(self.num_agents)]
            kl = [kl_div(self.hists[i], self.hists[i+1]) for i in range(len(self.hists)-1)]
            
            for n in range(self.num_agents):
                # Penalize for number of nodes allocated and similarity of histograms
                self.rewards[self.agents[n]] = -(np.ma.masked_invalid(kl).sum() +
                                                  len(np.where(self.alloc == None)[0])/self.num_nodes) 
                
        # The truncations dictionary must be updated for all players.
        self.truncations = {
            agent: self.num_steps >= 1000 for agent in self.agents # NOT USED
        }

        # observe the current state
        for i in self.agents:
            self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
            self.graph.nodes[action].update({'cluster': i}) # For the sake of plotting

        #else:
            # necessary so that observe() returns a reasonable observation at all times.
            #self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            # no rewards are allocated until both players give an action
            #self._clear_rewards()

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
        plt.savefig(r'src\rl_pipeline\SKA_allocation.png', bbox_inches='tight')

        if self.hists != None:
            fig, axes = plt.subplots(1, self.num_agents, figsize=(15, 5))
            for i in range(self.num_agents):
                axes[i].bar(self.bin_centers, self.hists[i], align='edge')

            fig.savefig(r'src\rl_pipeline\SKA_histograms.png', bbox_inches='tight')
        
        return

    # def _render_frame(self, colours):
    #     """
    #     Renders the environment. In human mode, it can print to terminal, open
    #     up a graphical window, or open up some other display that a human can see and understand.
    #     """
    #     if self.render_mode is None:
    #         gym.logger.warn(
    #             "You are calling render method without specifying any render mode."
    #         )
    #         return

    #     return 
