import gym
from gym import spaces
import networkx as nx
import numpy as np
import itertools
from math import dist
import matplotlib.pyplot as plt

class InterferometerEnv(gym.Env):
    def __init__(self, num_players=2, coordinate_file=r"src\dataset\ska_xy_coordinates.csv"):
        super(InterferometerEnv, self).__init__()

        # Graph representation: Fully connected graph with all antenna coordinates
        self.coordinates = np.genfromtxt(coordinate_file, delimiter=',')
        self.graph = nx.Graph()
        self.graph.add_nodes_from([(i, {'coordinates': self.coordinates[i], 
                                        'cluster': 0}) for i in range(len(self.coordinates))])
        self.graph.add_edges_from(itertools.combinations(self.graph.nodes, 2))
        
        self.current_step = 0
        self.num_players = num_players
        # Initialize agent collections
        self.agent_collections = {i: set() for i in range(self.num_players)}
        #num_baselines = (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1)) / 2

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(self.graph.number_of_nodes)
        #spaces.Discrete(self.graph.number_of_nodes() -
                                #(self.num_players * self.current_step)) # Number of unallocated nodes
        self.observation_space = gym.spaces.Dict({
            spaces.Dict({'subarray_configuration': self.graph.nodes['cluster'],
                            'baselines': np.array([[dist(self.graph[i]['coordinates'], 
                            self.graph[j]['coordinates']) for i in range(self.num_nodes)] 
                                        for j in range(self.num_nodes)])})
        })


    def step(self, actions):
        # Take action and update the environment. Check termination condition
        self.current_step += 1
        done = (self.current_step == self.num_nodes//self.num_players)

        obs_n = [self._get_obs(i) for i in range(self.num_players)] 
        rew_n = [self._get_reward(i) for i in range(self.num_players)]

        for n in range(self.num_players):
            self.graph.nodes[actions[n]].update({'cluster': n})

        return obs_n, rew_n, done, {}


    def reset(self):
        # Reset the environment
        self.current_step = 0
        self.agent_collections = {i: set() for i in range(self.num_agents)}
        observations = {i: self._get_obs(i) for i in range(self.num_agents)}
        [self.graph.nodes[i].update({'cluster': 0}) for i in range(self.graph.number_of_nodes)] # Unallocate all nodes

        return observations

    def _get_obs(self, agent_id): # TEST, just pick random nodes without regard for baseline
        # Get observation of a particular agent
        last_selected_node = max(self.agent_collections[agent_id]) if self.agent_collections[agent_id] else None
        return {'last_selected_node': self.graph.nodes[last_selected_node] if last_selected_node is not None else None}
        #return spaces.Dict({'subarray_configuration': self.graph.nodes['cluster'],
         #                               'baselines': np.array([[dist(self.graph[i]['coordinates'], 
          #                               self.graph[j]['coordinates']) for i in range(self.num_nodes)] 
           #                              for j in range(self.num_nodes)])})
    
    def _get_reward(self, agent_id): # TEST REWARD
        return -len(np.where(self.graph.nodes['cluster']==0))  # TEST Negative reward for each remaining unallocated node

    def render(self):
        # Implement visualization if needed
        colours = []
        color = iter(plt.cm.rainbow(np.linspace(0, 1, self.num_players)))
        for i in range(self.num_players):
            colours.append(next(color))
 
        plt.plot(self.coordinates[:, 0], self.coordinates[:, 1], 'o')

InterferometerEnv()