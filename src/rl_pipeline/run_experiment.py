from ska_env import InterferometerEnv
import wandb
import torch
from agents import PPOAgent, SGD_Agent
from sgd import train_SGD_agents

if __name__ == "__main__":
    num_agents = 2
    num_episodes = 15
    episode_length = 100
    env = InterferometerEnv(1,1, num_agents=num_agents)

    run = wandb.init(
    # Set the project where this run will be logged
    project="ska_RL",
    # Track hyperparameters and run metadata
    config={
        "num_agents": num_agents,
        "episodes": num_episodes,
    },
    )
    agents = [SGD_Agent(num_actions=env.num_nodes,
                        agent_id=env.possible_agents[i]) for i in range(num_agents)]
    train_SGD_agents(env, agents, num_episodes=num_episodes, episode_length=episode_length,
                      agent_ids=env.possible_agents)
