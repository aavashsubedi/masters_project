from ska_env import InterferometerEnv
import wandb
import torch
import hydra
from rl_utils import setup_wandb
from agents import *
from train_agents import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REGIMES = {'Random': [SGD, train_SGD, 1],
           'SGD': [SGD, train_SGD, 0.1],
           'LOLA': [LOLA, train_LOLA, None],
           'PPO': [PPO, train_PPO, None]}

@hydra.main(version_base='1.3', config_path="config/",
             config_name="cfg.yaml")
def main(cfg):
    setup_wandb(cfg)
    num_agents = cfg.num_agents
    num_episodes = cfg.episodes
    episode_length = cfg.episode_length
    try:
        agent, train, exploration = REGIMES[cfg.agent_type]
    except IndexError:
        print("Agent type not valid. Choose from 'Random','SGD','LOLA','PPO'")

    env = InterferometerEnv(cfg.target_sensitivity, cfg.target_resolution,
                             num_agents=num_agents)

    agents = [agent(num_actions=env.num_nodes, learning_rate=cfg.learning_rate,
                        agent_id=env.possible_agents[i], exploration=exploration) for i in range(num_agents)]
    train(env, agents, num_episodes=num_episodes, episode_length=episode_length,
                      agent_ids=env.possible_agents)

main()