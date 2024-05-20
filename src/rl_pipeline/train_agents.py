import wandb
import torch
from tqdm import tqdm
import os
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from agents import SPGActor, SPGCritic
from agents import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_SPG(env, num_episodes, episode_length, actor_lr, critic_lr,
                actor_lr_decay_step, actor_lr_decay_rate, 
                critic_lr_decay_step, critic_lr_decay_rate,
                epsilon, epsilon_step, epsilon_decay,
                train_step, batch_size, max_grad_norm,
                save_dir="saved_models/"):
    actor = SPGActor(state_dim=env.num_nodes).to(device)
    critic = SPGCritic(state_dim=env.num_nodes).to(device)

    wandb.watch(actor, log='all', log_freq=10)
    wandb.watch(critic, log='all', log_freq=10)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    critic_loss = torch.nn.MSELoss().to(device)
    critic_aux_loss = torch.nn.MSELoss().to(device)

    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
        range(actor_lr_decay_step, actor_lr_decay_step * 1000,
            actor_lr_decay_step), gamma=actor_lr_decay_rate)
    critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
        range(critic_lr_decay_step, critic_lr_decay_step * 1000,
            critic_lr_decay_step), gamma=critic_lr_decay_rate)

    # Instantiate replay buffer
    observation_shape = [env.num_nodes, 1]
    replay_buffer = Memory(1000, action_shape=[env.num_nodes, env.num_nodes], 
            observation_shape=observation_shape)
    
    for episode in tqdm(range(num_episodes)):

        states, info = env.reset()
        done = False
        total_reward = 0

        for i in tqdm(range(episode_length)):
            psi, action = actor(states)
            #print(action)

            # Epsilon exploration
            if np.random.rand() < 1: #epsilon:
            # Add noise in the form of 2-exchange neighborhoods
                for r in range(2):
                    # randomly choose two row idxs
                    idxs = np.random.randint(0, env.num_nodes, size=2)
                    # swap the two rows
                    tmp = action[:, idxs[0]].clone()
                    tmp2 = action[:, idxs[1]].clone()
                    tmp3 = psi[:, idxs[0]].clone()
                    tmp4 = psi[:, idxs[1]].clone()
                    action[:, idxs[0]] = tmp2
                    action[:, idxs[1]] = tmp
                    psi[:, idxs[0]] = tmp4
                    psi[:, idxs[1]] = tmp3
            if train_step > 0 and epsilon > 0.01:
                epsilon -= epsilon_decay

            #import pdb; pdb.set_trace()
            #wandb.log({'States': states.cpu().numpy()})
                       
            # apply the permutation to the input
            #solutions = torch.matmul(torch.transpose(states, 1, 2), action)

            next_state, reward, done, _ = env.step(action)
            states = next_state
            total_reward += reward
            replay_buffer.append(states.data, action.data.byte(), psi.data, torch.tensor(reward).unsqueeze(0).data)

            if (replay_buffer.nb_entries > batch_size) and replay_buffer.nb_entries>2:
                s_batch, a_batch, psi_batch, r_batch = replay_buffer.sample(batch_size)
                s_batch = torch.cat((s_batch, s_batch), 0)
                a_batch = torch.cat((a_batch, a_batch), 0).float()
                psi_batch = torch.cat((psi_batch, psi_batch), 0)
                targets = torch.cat((r_batch, r_batch), 0)

                # N.B. We use the actions from the replay buffer to update the critic
                # a_batch_t are the hard permutations
                #import pdb; pdb.set_trace()
                hard_Q = critic(s_batch, a_batch).squeeze(2)
                critic_out = critic_loss(hard_Q, targets)

                soft_Q = critic(s_batch, psi_batch).squeeze(2)
                critic_aux_out = critic_aux_loss(soft_Q, hard_Q.detach())
                critic_optim.zero_grad()
                (critic_out + critic_aux_out).backward()

                # clip gradient norms
                torch.nn.utils.clip_grad_norm_(critic.parameters(),
                    max_grad_norm, norm_type=2)
                critic_optim.step()
                critic_scheduler.step()                 
                
                critic_optim.zero_grad()                
                actor_optim.zero_grad()
                soft_action, _ = actor(s_batch)
                # N.B. we use the action just computed from the actor net here, which 
                # will be used to compute the actor gradients
                # compute gradient of critic network w.r.t. actions, grad Q_a(s,a)
                soft_critic_out = critic(s_batch, soft_action).squeeze(2).mean()
                actor_loss = -soft_critic_out
                actor_loss.backward()

                # clip gradient norms
                torch.nn.utils.clip_grad_norm_(actor.parameters(),
                    max_grad_norm, norm_type=2)

                actor_optim.step()
                actor_scheduler.step()

            #if i % 50 == 0:
             #   torch.save(actor, os.path.join(save_dir, 'actor-epoch-{}.pt'.format(i+1)))
              #  torch.save(critic, os.path.join(save_dir, 'critic-epoch-{}.pt'.format(i+1)))
        env.close()
        log = total_reward/episode_length
        wandb.log({'Episode Reward': log})
        print(f"Episode {episode+1}/{num_episodes}, Total Rewards: {total_reward.item()/episode_length}\n")


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

        wandb.log({'{0} Episode Reward'.format(agent_ids[i]): total_rewards[i]/episode_length for i in range(num_agents)})
        print(f"Episode {episode+1}/{num_episodes}, Total Rewards: {total_rewards/episode_length}")


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
                next_state, reward, done[i], _ = env.step(action)
                actions.append(action)

                agents[i].update_weights(actions[i], actions[i-1], reward[agent_ids[i]], reward[agent_ids[i-1]])
                states[agent_ids[i]] = next_state
                total_rewards[i] += list(reward.values())[i]
        env.close()

        wandb.log({'{0} Episode Reward'.format(agent_ids[i]): total_rewards[i] for i in range(num_agents)})
        print(f"Episode {episode+1}/{num_episodes}, Total Rewards: {total_rewards}")


def train_PPO(env, agents, num_episodes, episode_length, agent_ids):
    num_agents = len(agents)
    
    for episode in range(num_episodes):
        #import pdb; pdb.set_trace()
        states, info = env.reset() # states is a dict
        done = False
        episode_rewards = [0] * num_agents
        for _ in range(episode_length):
            next_states = []
            for i, agent in enumerate(agents):
                dist, action = agent.select_action(states[agent_ids[i]])
                next_state, rewards, done, _ = env.step(action)
                action = torch.tensor(action, device=device)
                episode_rewards = [x + y for x, y in zip(episode_rewards, rewards.values())]
                next_states.append(next_state)

                agent.update((states[agent_ids[i]], action, 
                              dist.log_prob(action),
                               rewards[agent_ids[i]], None, done))
            states = next_states
        env.close()

        wandb.log({'{0} Episode Reward'.format(agent_ids[i]): episode_rewards[i] for i in range(num_agents)})
        print(f"Episode {episode}: Total rewards: {episode_rewards}")


def train_DDPG(env, agents, num_episodes, episode_length, agent_ids):
    num_agents = len(agents)
    
    for episode in range(num_episodes):
        #import pdb; pdb.set_trace()
        states, info = env.reset() # states is a dict
        done = False
        episode_rewards = [0] * num_agents
        for _ in range(episode_length):
            next_states = []
            for i, agent in enumerate(agents):
                action = agent.select_action(states[agent_ids[i]])
                next_state, rewards, done, _ = env.step(action)
                action = torch.tensor(action, device=device)
                episode_rewards = [x + y for x, y in zip(episode_rewards, rewards.values())]
                next_states.append(next_state)

                agent.update()
            states = next_states
        env.close()

        wandb.log({'{0} Episode Reward'.format(agent_ids[i]): episode_rewards[i] for i in range(num_agents)})
        print(f"Episode {episode}: Total rewards: {episode_rewards}")


def train_DQN(env, agents, num_episodes, episode_length, agent_ids):
    num_agents = len(agents)
    
    for episode in range(num_episodes):
        states, info = env.reset() # states is a dict
        done = False
        episode_rewards = [0] * num_agents

        for _ in range(episode_length):
            for i, agent in enumerate(agents):
                action = agent.select_action(states[agent_ids[i]])
                next_state, rewards, done, _ = env.step(action)

                agent.replay_buffer.push(states[agent_ids[i]], action, 
                                        next_state[agent_ids[i]], rewards[agent_ids[i]], done)
                agent.update()
                agent.decay_epsilon()
                states[agent_ids[i]] = next_state
                episode_rewards = [x + y for x, y in zip(episode_rewards, rewards.values())]
        env.close()

        wandb.log({'{0} Episode Reward'.format(agent_ids[i]): episode_rewards[i] for i in range(num_agents)})
        print(f"Episode {episode}: Total rewards: {episode_rewards}")
