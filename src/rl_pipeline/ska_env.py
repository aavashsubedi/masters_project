import gym
from gym import spaces
import networkx as nx
import numpy as np
import itertools
from math import dist
import matplotlib.pyplot as plt

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