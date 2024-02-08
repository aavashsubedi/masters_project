import gym
from gym import spaces
from gymnasium.spaces import Dict, Box, Discrete, MultiDiscrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

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
    env = raw_env(render_mode=internal_render_mode)
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
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, num_nodes=197, num_agents=2, render_mode=None):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        # Creating graph. Keeping this for the sake of plotting
        self.coordinates = np.genfromtxt(coordinate_file, delimiter=',')
        self.graph = nx.Graph()
        self.graph.add_nodes_from([(i, {'coordinates': self.coordinates[i], 
                                        'cluster': None}) for i in range(len(self.coordinates))])
        self.graph.add_edges_from(itertools.combinations(self.graph.nodes, 2))

        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.possible_agents = ["player_" + str(r) for r in range(self.num_agents)]

        # Mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self._action_spaces = {agent: Discrete(self.graph.number_of_nodes()) for agent in self.possible_agents}
        self._observation_spaces = {
            agent: MultiDiscrete([n for n in range(self.num_nodes)]) for agent in self.possible_agents
        } # This is the allocation of all nodes, which can be allocated to any of n agents
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    #@functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([n for n in range(self.num_nodes)])

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(self.graph.number_of_nodes())

    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: NONE for agent in self.agents}
        self.observations = {agent: NONE for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

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
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # stores action of current agent
        self.state[self.agent_selection] = action

        # rewards for all agents are placed in the .rewards dictionary
        # Could introduce baseline weights here!!!! Start with uniform for now
        hists = [np.histogram(baseline_dists(np.where(self.coordinates[action[i]])), bins=[n*10 for n in range(1,16)], 
                                density=True)[0] for i in range(self.num_agents)]
        kl = [kl_div(hists[i], hists[i+1]) for i in range(len(hists))]
        
        for n in range(self.num_agents):
            self.rewards[self.agents[n]] = -sum(kl)

        self.num_moves += 1
        # The truncations dictionary must be updated for all players.
        self.truncations = {
            agent: self.num_moves >= NUM_ITERS for agent in self.agents
        }

        # observe the current state
        for i in self.agents:
            self.observations[i] = self.state[self.agents[1 - self.agent_name_mapping[i]]]
            self.graph.nodes[action[i]].update({'cluster': i}) # For the sake of plotting

        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = NONE
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations


    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        colours = []
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.num_agents)))
        for i in range(self.num_agents):
            colours.append(next(color))

        for i in range(self.num_agents):
            for n in range(self.num_nodes):
                if self.graph.nodes[n]['cluster'] == i:
                    plt.plot(self.coordinates[n, 0], self.coordinates[n, 1], '.', color=colours[i], 
                                label='Agent {}'.format(i+1))
            
        plt.legend()
        plt.savefig(r'src\rl_pipeline\SKA_allocation.png', bbox_inches='tight')

        return 

"""
class SKAEnv(gym.Env):
    def __init__(self, num_agents=2, coordinate_file=r"src\dataset\ska_xy_coordinates.csv"):
        super(SKAEnv, self).__init__()

        # Graph representation: Fully connected graph with all antenna coordinates
        self.coordinates = np.genfromtxt(coordinate_file, delimiter=',')
        self.graph = nx.Graph()
        self.graph.add_nodes_from([(i, {'coordinates': self.coordinates[i], 
                                        'cluster': None}) for i in range(len(self.coordinates))])
        self.graph.add_edges_from(itertools.combinations(self.graph.nodes, 2))
        
        self.current_step = 0
        self.num_agents = num_agents
        self.num_nodes = self.graph.number_of_nodes()
        # Initialize agent collections
        self.agent_collections = {i: set() for i in range(self.num_agents)}
        #num_baselines = (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)) / 2

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(self.num_nodes)
        #spaces.Discrete(self.graph.number_of_nodes() -
                                #(self.num_agents * self.current_step)) # Number of unallocated nodes
        self.observation_space = gym.spaces.Dict({'subarray_configuration': gym.Space([self.graph.nodes[i]['cluster'] for i in range(self.num_nodes)]),
                            'baselines': gym.Space(np.array([[dist(self.graph.nodes[i]['coordinates'], self.graph.nodes[j]['coordinates']) for i in range(self.num_nodes)] for j in range(self.num_nodes)]))
        })


    def step(self, actions):
        # Take action and update the environment. Check termination condition
        self.current_step += 1
        if self.current_step <=200:
            done = (self.current_step == self.num_nodes//self.num_agents)
        else:
            done = True

        obs_n = [self._get_obs(i) for i in range(self.num_agents)] 
        rew_n = [self._get_reward(i) for i in range(self.num_agents)]

        for n in range(self.num_agents):
            self.graph.nodes[actions[n]].update({'cluster': n})

        return obs_n, rew_n, done, {}


    def reset(self):
        # Reset the environment
        self.current_step = 0
        self.agent_collections = {i: set() for i in range(self.num_agents)}
        observations = {i: self._get_obs(i) for i in range(self.num_agents)}
        [self.graph.nodes[i].update({'cluster': None}) for i in range(self.num_nodes)] # Unallocate all nodes

        return observations

    def _get_obs(self, agent_id): # TEST, just pick random nodes without regard for baseline
        # Get observation of a particular agent
        last_selected_node = max(self.agent_collections[agent_id]) if self.agent_collections[agent_id] else None
        return {'last_selected_node': last_selected_node,} #{'subarray_configuration': [self.graph.nodes[i]['cluster'] for i in range(self.num_nodes)],}
                                        #'baselines': np.array([[dist(self.graph[i]['coordinates'], 
                                         #self.graph[j]['coordinates']) for i in range(self.num_nodes)] 
                                         #for j in range(self.num_nodes)])}
    
    #{'last_selected_node': self.graph.nodes[last_selected_node] if last_selected_node is not None else None}

    
    def _get_reward(self, agent_id): # TEST REWARD
        return -len(np.where([self.graph.nodes[i]['cluster'] for i in range(self.num_nodes)]==0))  # TEST Negative reward for each remaining unallocated node

    def render(self):
        # Implement visualization if needed
        colours = []
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.num_agents)))
        for i in range(self.num_agents):
            colours.append(next(color))

        for i in range(self.num_agents):
            for n in range(self.num_nodes):
                if self.graph.nodes[n]['cluster'] == i:
                    plt.plot(self.coordinates[n, 0], self.coordinates[n, 1], '.', color=colours[i], 
                                label='Agent {}'.format(i+1))
            
        #plt.legend()

        plt.savefig(r'src\rl_pipeline\SKA_allocation.png', bbox_inches='tight')
        """