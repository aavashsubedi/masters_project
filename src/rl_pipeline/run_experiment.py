from ska_env import InterferometerEnv
import wandb
import torch
import hydra
from rl_utils import setup_wandb
from agents import PPO, SGD, LOLA
from rl_pipeline.train_agent import train_agents

@hydra.main(version_base='1.3', config_path="config/",
             config_name="cfg.yaml")
def main(cfg):
    setup_wandb(cfg)
    num_agents = 2
    num_episodes = 15
    episode_length = 100

    env = InterferometerEnv(cfg.target_sensitivity, cfg.target_resolution,
                             num_agents=num_agents)

    #run = wandb.init(
    # # Set the project where this run will be logged
    # project="ska_RL",
    # # Track hyperparameters and run metadata
    # config={
    #     "num_agents": num_agents,
    #     "episodes": num_episodes,
    # },
    # )
    agents = [SGDAgent(num_actions=env.num_nodes, learning_rate=cfg.learning_rate,
                        agent_id=env.possible_agents[i]) for i in range(num_agents)]
    train_agents(env, agents, num_episodes=num_episodes, episode_length=episode_length,
                      agent_ids=env.possible_agents)

main()