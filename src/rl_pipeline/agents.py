import numpy as np
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from rl_utils import *
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch.autograd import Variable
from pathos.multiprocessing import ProcessingPool as Pool
from spg.util import parallel_matching

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""All agents need a select_action() and update() function"""

class SPGActor(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, sinkhorn_iters=5, sinkhorn_tau=1):
        super(SPGActor, self).__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.embedding = nn.Linear(1, hidden_dim) # Embed state to high dim
        self.sinkhorn = Sinkhorn(state_dim, sinkhorn_iters, sinkhorn_tau)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(self.hidden_dim, state_dim)
        self.round = linear_assignment
        self.init_h = torch.zeros(1, hidden_dim, device=device)

    def forward(self, x): # State is allocation 
        """
        x is [batch_size, state_dim, 1]
        """
        if x.dtype !=torch.float32:
            x = torch.tensor(x, dtype=torch.float32, device=device)
        #import pdb; pdb.set_trace()
        batch_size = x.size()[0]
        # Embedding state to high dimension
        #import pdb; pdb.set_trace()
        x = F.leaky_relu(self.embedding(x))
        x = torch.transpose(x, 0, 1)
        
        # x size [1, 128, 1]
        #import pdb; pdb.set_trace()
        # Init hidden state
        init_h = self.init_h.unsqueeze(1).repeat(1, batch_size, 1)
        h_last, _ = self.gru(x, init_h)
        #import pdb; pdb.set_trace()
        # h_last should be [state_dim, batch_size, decoder_dim]
        x = torch.transpose(h_last, 0, 1)
        # transform to [batch_size, state_dim, state_dim]
        #import pdb; pdb.set_trace()
        M = self.fc1(x)
        psi = self.sinkhorn(M) # Doubly-stochastic
        #import pdb; pdb.set_trace()

        # Round doubly-stochastic psi to permutation matrix
        # Hungarian algorithm
        batch = psi.data.cpu().numpy()
        perms = []
        for i in range(batch_size):
            perm = torch.zeros(self.state_dim, self.state_dim)                
            # Contains indices of row, then column in 2nd array   
            matching = self.round(-batch[i]) 
            perm[matching[0], matching[1]] = 1
            perms.append(perm)
        perms = torch.stack(perms).to(device)

        return psi, perms


class SPGCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=197):
        super(SPGCritic, self).__init__()
        self.embeddingX = nn.Linear(1, hidden_dim) # Embed state to high dim
        self.embeddingP = nn.Linear(state_dim, hidden_dim) # Embed action
        self.combine = nn.Linear(hidden_dim, hidden_dim) # Fuse embeddings with element-wise sum
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.init_h = torch.zeros(1, hidden_dim, device=device)

        self.fc1 = nn.Linear(hidden_dim, 1)
        self.fc2 = nn.Linear(state_dim, 1)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(state_dim)
        self.bn2 = nn.BatchNorm1d(state_dim)
        self.bn3 = nn.BatchNorm1d(state_dim)

    def forward(self, x, p):
        batch_size = x.size()[0]
        import pdb; pdb.set_trace
        x = F.leaky_relu(self.bn1(self.embeddingX(x)))
        p = F.leaky_relu(self.bn2(self.embeddingP(p)))
        xp = F.leaky_relu(self.bn3(self.combine(x + p)))
        x = torch.transpose(xp, 0, 1)
        init_h = self.init_h.unsqueeze(1).repeat(1, batch_size, 1)
        h_last, hidden_state = self.gru(x, init_h)

        # h_last should be [n_nodes, batch_size, decoder_dim]
        x = torch.transpose(h_last, 0, 1)
        x = F.leaky_relu(self.fc3(x))
        out = self.fc1(x)
        out = self.fc2(torch.transpose(out, 1, 2))
        # out is [batch_size, 1, 1]
        return out 
    

class SPG(nn.Module):
    def __init__(self, num_nodes, actor_lr, critic_lr,
                actor_lr_decay_step, actor_lr_decay_rate,
                critic_lr_decay_step, critic_lr_decay_rate):
        super(SPG, self).__init__()
        self.actor = SPGActor(state_dim=num_nodes).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = SPGCritic(state_dim=num_nodes).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_loss = torch.nn.MSELoss().to(device)
        self.critic_aux_loss = torch.nn.MSELoss().to(device)

        self.actor_scheduler = lr_scheduler.MultiStepLR(self.actor_optim,
            range(actor_lr_decay_step, actor_lr_decay_step * 1000,
                actor_lr_decay_step), gamma=actor_lr_decay_rate)
        self.critic_scheduler = lr_scheduler.MultiStepLR(self.critic_optim,
            range(critic_lr_decay_step, critic_lr_decay_step * 1000,
                critic_lr_decay_step), gamma=critic_lr_decay_rate)

        # Instantiate replay buffer
        observation_shape = [num_nodes, 1]
        self.replay_buffer = Memory(10000, action_shape=[num_nodes, num_nodes], 
                observation_shape=[num_nodes, 1])
        

    def select_action(self, state, noise_clip=0.5):
        try:
            state = torch.clone(state, dtype=torch.float, device=device).detach().unsqueeze(0)
        except TypeError:
            state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        action += noise_clip * np.random.normal(0, 1, size=action.shape)

        return np.clip(action, -1, 1)
        

    def update(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Compute Q(s_t, mu(s_t)=a_t)
        # size is [batch_size, 1]
        # N.B. We use the actions from the replay buffer to update the critic
        # a_batch_t are the hard permutations
        hard_Q = self.critic(s_batch, a_batch).squeeze(2)
        critic_out = critic_loss(hard_Q, targets)
        if not args['disable_critic_aux_loss']:
            soft_Q = critic(s_batch, psi_batch).squeeze(2)
            critic_aux_out = critic_aux_loss(soft_Q, hard_Q.detach())
            critic_optim.zero_grad()
            (critic_out + critic_aux_out).backward()
        else:
            critic_optim.zero_grad()
            critic_out.backward() 
        # clip gradient norms
        torch.nn.utils.clip_grad_norm(critic.parameters(),
            args['max_grad_norm'], norm_type=2)
        critic_optim.step()
        critic_scheduler.step()                 
        
        critic_optim.zero_grad()                
        actor_optim.zero_grad()
        soft_action, _ = actor(s_batch, do_round=False)
        # N.B. we use the action just computed from the actor net here, which 
        # will be used to compute the actor gradients
        # compute gradient of critic network w.r.t. actions, grad Q_a(s,a)
        soft_critic_out = critic(s_batch, soft_action).squeeze(2).mean()
        actor_loss = -soft_critic_out
        actor_loss.backward()

        # clip gradient norms
        torch.nn.utils.clip_grad_norm(actor.parameters(),
            args['max_grad_norm'], norm_type=2)

        actor_optim.step()
        actor_scheduler.step()


        return None


################################
# The following agents were for the multi-agent environment which I have ruined and turned into a single-agent env.
# They may not work anymore

class SGD:
    def __init__(self, num_actions, agent_id, learning_rate, batch_size, exploration=0.1):
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.weights = torch.ones(num_actions, device=device) * 0 # Optimistic initialisation
    
    def select_action(self, state):
        # Select action using epsilon-greedy policy
        if np.random.rand() < self.exploration: # Exploration percentage
            wandb.log({'{0} weights'.format(self.agent_id): self.weights})
            return np.random.choice(self.num_actions)
        else:
            wandb.log({'{0} weights'.format(self.agent_id): self.weights})
            return torch.argmax(self.weights)

    def update_weights(self, state, action, reward):
        # Update weights using stochastic gradient descent
        self.weights[action] += self.learning_rate * reward[self.agent_id]


class LOLA(nn.Module):
    def __init__(self, num_actions, agent_id, learning_rate, batch_size, exploration): # exploration not used.
        self.agent_id = agent_id
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.Q = torch.zeros((num_actions, num_actions), requires_grad=True) # Q-values for joint actions
        self.weights = (torch.ones(num_actions, num_actions, device=device) / num_actions).squeeze() # Stochastic policy
    
    def select_action(self, opponent_action):
        if opponent_action == None:
            opponent_action = 0
        # Convert Q-values to probabilities using softmax
        #probs = self.softmax(self.Q[:, opponent_action])
        policy = self.weights.unsqueeze(0)
        # Sample action from the probability distribution
        wandb.log({'{0} weights'.format(self.agent_id): self.weights})

        return torch.multinomial(policy[:, :, opponent_action], 1).item() # Without replacement

    def update_weights(self, own_action, opponent_action, own_reward, opponent_reward):
        # Update weights using policy gradient
        gradient = torch.zeros((self.num_actions, self.num_actions), device=device)
        gradient[own_action, opponent_action] = 1

        # Update Q-values using gradient descent
        q_values = self.Q[own_action, opponent_action]
        loss = -(own_reward + opponent_reward - q_values)
        loss.backward()
        
        with torch.no_grad():
            # Update Q-values
            self.Q.data -= self.learning_rate * self.Q.grad
            # Reset gradients
            self.Q.grad.zero_()
            # Update policy using softmax function
            self.weights = F.softmax(self.Q, dim=0)

        return None


class PPO:
    def __init__(self, num_actions, num_states, agent_id, learning_rate, batch_size,
                 gamma=0.99, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01):
        self.policy_net = PolicyNetwork(num_actions).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.num_actions = num_actions
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.agent_id = agent_id

    def select_action(self, state):
        state_tensor = torch.clone(state, dtype=torch.float, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = self.policy_net(state_tensor)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return dist, action.item()

    def update(self, rollout):
        states, actions, old_log_probs, rewards, next_states, dones = rollout
        states = torch.clone(states, dtype=torch.float, device=device) # Change type and location
        returns = self._compute_returns(rewards)
        
        for _ in range(400):  # PPO epoch
            #import pdb; pdb.set_trace()
            logits = self.policy_net(states)
            
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            wandb.log({'{0} weights'.format(self.agent_id): log_probs})
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * returns
            actor_loss = -torch.min(surr1, surr2).mean()
            
            values = self.policy_net(states) #.gather(1, actions.unsqueeze(1)).squeeze(1)
            value_loss = nn.MSELoss()(values, returns)

            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            #import pdb; pdb.set_trace()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _compute_returns(self, rewards):
        returns = []
        R = 0
        #for r in reversed(rewards):
        R = rewards + self.gamma * R
        returns.insert(0, R)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)  # normalize returns

        return torch.tensor(rewards, dtype=torch.float, device=device)
    

class DDPG:
    def __init__(self, num_actions, num_states, learning_rate, batch_size, agent_id, 
                 hidden_dim_actor=64, hidden_dim_critic=64,
                 gamma=0.99, tau=0.005, buffer_capacity=10000):
        self.actor_lr = learning_rate
        self.actor = Actor(num_states, num_actions, hidden_dim_actor).to(device)
        self.actor_target = Actor(num_states, num_actions, hidden_dim_actor).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic_lr = learning_rate
        self.critic = Critic(num_states, num_actions, hidden_dim_critic).to(device)
        self.critic_target = Critic(num_states, num_actions, hidden_dim_critic).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

    def select_action(self, state, noise_clip=0.5):
        try:
            state = torch.clone(state, dtype=torch.float, device=device).detach().unsqueeze(0)
        except TypeError:
            state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        action += noise_clip * np.random.normal(0, 1, size=action.shape)

        return np.clip(action, -1, 1)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Update critic
        next_actions = self.actor_target(next_states)
        target_q = rewards + self.gamma * (1 - dones) * self.critic_target(next_states, next_actions)
        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._update_target_networks()

        return None

    def _update_target_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class DQN:
    def __init__(self, num_states, num_actions, learning_rate, agent_id, 
                gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, 
                epsilon_decay=0.995, target_update=10, batch_size=64):
        self.q_network = QNetwork(num_states, num_actions).to(device)
        self.target_network = QNetwork(num_states, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.steps_done = 0
        self.num_actions = num_actions
        self.agent_id = agent_id

    def select_action(self, state):
        try:
            state = torch.clone(state, dtype=torch.float, device=device).detach().unsqueeze(0)
        except TypeError:
            state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        
        self.steps_done += 1
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax().item()
        else:
            action = np.random.randint(self.num_actions)
        return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=device)
        next_state_batch = torch.cat(batch.next_state)
        import pdb; pdb.set_trace()
        done_mask = [ele[self.agent_id] for ele in batch.done]

        import pdb; pdb.set_trace()
        current_q_values = self.q_network(state_batch).gather(1, action_batch)
        import pdb; pdb.set_trace()
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - all(done_mask)) * self.gamma * next_q_values

        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)