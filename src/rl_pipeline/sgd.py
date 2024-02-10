def train_SGD_agents(env, agents, num_episodes, agent_ids):
    num_agents = len(agents)
    
    for episode in range(num_episodes):
        states, info = env.reset()
        done = [False] * num_agents
        total_rewards = [0] * num_agents

        while not all(done):
            for i in range(num_agents):
                for step in range(env.num_steps):
                    action = agents[i].select_action(states[agent_ids[i]]) # Problem with states
                    next_state, reward, done[i], _ = env.step(action)
                    agents[i].update_weights(states[agent_ids[i]], action, reward)
                    states[agent_ids[i]] = next_state
                    total_rewards[i] += list(reward.values())[i]
                
                    if step % 20 == 0:
                        env.render()
        env.close()

        print(f"Episode {episode+1}/{num_episodes}, Total Rewards: {total_rewards}")
