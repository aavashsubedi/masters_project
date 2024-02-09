from ska_env import InterferometerEnv
import numpy as np
import pettingzoo
from pettingzoo.test import api_test, seed_test, render_test


env = InterferometerEnv(render_mode="human")
env.reset(seed=42)

## TESTS:
#api_test(InterferometerEnv(), num_cycles=1000, verbose_progress=False)
#seed_test(InterferometerEnv, num_cycles=10)
#render_test(InterferometerEnv)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        if "action_mask" in info:
            mask = info["action_mask"]
        elif isinstance(observation, dict) and "action_mask" in observation:
            mask = observation["action_mask"]
        else:
            mask = None
        action = env.action_space(agent).sample(mask)


    # if termination or truncation:
    #     action = None
    # else:
    #     # this is where you would insert your policy
    #     action = env.action_space(agent).sample()

    env.step(action)
    env.render()
env.close()


"""
# Create the SKA environment
env = InterferometerEnv()

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
"""