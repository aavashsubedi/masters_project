from ska_env import InterferometerEnv
import wandb
import torch
import hydra
from rl_utils import setup_wandb
from agents import *
from train_agents import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base='1.3', config_path="config/",
             config_name="cfg.yaml")
def main(cfg):
    setup_wandb(cfg)
    num_agents = cfg.num_agents
    num_episodes = cfg.episodes
    episode_length = cfg.episode_length
    agent = SGD
    train = train_SGD

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
    agents = [agent(num_actions=env.num_nodes, learning_rate=cfg.learning_rate,
                        agent_id=env.possible_agents[i], exploration=1) for i in range(num_agents)]
    train(env, agents, num_episodes=num_episodes, episode_length=episode_length,
                      agent_ids=env.possible_agents)

main()