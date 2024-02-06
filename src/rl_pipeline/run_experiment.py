import gym
from ska_env import SKAEnv
import numpy as np

# Create the SKA environment
env = SKAEnv()

# Reset the environment to get initial observations
observations = env.reset()

# Define a simple random agent
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self):
        return self.action_space.sample()
    
# Define a simple SGD learner
class SGDAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, available_nodes):
        return np.random.choice(available_nodes)

# Create an SGD learner for the player
learner = SGDAgent(env.action_space)

# Create random agents for each player
agents = {i: RandomAgent(env.action_space) for i in range(env.num_agents)}

# Run a few episodes
for episode in range(3):
    print(f"Episode {episode + 1}")

    while True:
        # Each agent chooses an action
        actions = {i: agents[i].choose_action() for i in range(env.num_agents)}

        # Take a step in the environment
        observations, rewards, done, _ = env.step(actions)

        # Print the current state
        print("Observations:", observations)
        print("Rewards:", rewards)

        env.render()

        # Check if all agents are done
        if done:
            break
