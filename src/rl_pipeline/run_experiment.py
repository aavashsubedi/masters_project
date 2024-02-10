from ska_env import InterferometerEnv
import numpy as np
import pettingzoo
from pettingzoo.test import api_test, seed_test, render_test
import torch
import torch.optim as optim
import torch.nn as nn
from agents import PPOAgent, SGD_Agent
from rl_utils import batchify_obs, batchify, unbatchify
from sgd import train_SGD_agents

if __name__ == "__main__":
    num_agents = 2
    env = InterferometerEnv(num_agents=num_agents)
    agents = [SGD_Agent(num_actions=env.num_nodes, 
                        agent_id=env.possible_agents[i]) for i in range(num_agents)]
    train_SGD_agents(env, agents, num_episodes=10, agent_ids=env.possible_agents)
