import wandb
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_SGD(env, agents, num_episodes, episode_length, agent_ids):
    num_agents = len(agents)
    
    for episode in range(num_episodes):
        states, info = env.reset()
        done = [False] * num_agents
        total_rewards = [0] * num_agents

        while not all(done):
            for _ in range(episode_length):
                for i in range(num_agents):
                    action = agents[i].select_action(states[agent_ids[i]])
                    #print(action)
                    next_state, reward, done[i], _ = env.step(action)
                    agents[i].update_weights(states[agent_ids[i]], action, reward)
                    states[agent_ids[i]] = next_state
                    total_rewards[i] += list(reward.values())[i]
        env.close()

        wandb.log({'{0} Episode Reward'.format(agent_ids[i]): total_rewards[i] for i in range(num_agents)})
        print(f"Episode {episode+1}/{num_episodes}, Total Rewards: {total_rewards}")


def train_LOLA(env, agents, num_episodes, episode_length, agent_ids):
    num_agents = len(agents) # ONLY WORKS with 2 for now
    
    for episode in range(num_episodes):
        states, info = env.reset()
        done = [False] * num_agents
        total_rewards = [0] * num_agents

        for _ in range(episode_length):
            actions = [] # Track actions for each player so LOLA can update
            for i in range(num_agents):
                action = agents[i].select_action(states[agent_ids[i]])
                #print(action)
                next_state, reward, done[i], _ = env.step(action)
                actions.append(action)

                agents[i].update_weights(actions[i], actions[i-1], reward[agent_ids[i]], reward[agent_ids[i-1]])
                states[agent_ids[i]] = next_state
                total_rewards[i] += list(reward.values())[i]
        env.close()

        wandb.log({'{0} Episode Reward'.format(agent_ids[i]): total_rewards[i] for i in range(num_agents)})
        print(f"Episode {episode+1}/{num_episodes}, Total Rewards: {total_rewards}")
