import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import random
import torch.optim as optim

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed,  fcs1_units=512, fc2_units=256):
        
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
import random 
import numpy as np 
import torch 
from collections import namedtuple, deque
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:

    def __init__(self,seed, buffer_size):
        self.memory = deque(maxlen = buffer_size)
        self.experience = namedtuple("Experience", field_names = ["full_state","state","action","reward","full_next_state","next_state","done"])
        self.seed = random.seed(seed)

    def add(self,full_states, state,action,reward,full_next_state, next_state,done):
        e = self.experience(full_states, state,action,reward,full_next_state, next_state,done)
        self.memory.append(e)


    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)

        full_states = torch.from_numpy(np.array([e.full_state for e in experiences if e is not None])).float().to(device)
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(device)
        full_next_states = torch.from_numpy(np.array([e.full_next_state for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (full_states, states, actions, rewards, full_next_states, next_states, dones)

    def __len__(self):
        return len(self.memory)
    
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np 
import random

class Config:
    def __init__(self,state_size ,action_size,random_seed,n_agents,EPISODES_ROLLOUT = 300,noise_start = 1, noise_decay = 0.99 ,noise_end = 0.1 ,buffer_size = int(1e5),batch_size = 128, gamma = 0.99,lr_actor = 1e-4,lr_critic = 1e-3,update_every = 20,epoch = 10, tau = 1e-3):
        
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.n_agents = n_agents
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.LR_ACTOR = lr_actor
        self.LR_CRITIC = lr_critic
        self.UPDATE_EVERY = update_every
        self.EPOCH = epoch
        self.TAU = tau
        self.NOISE_START = noise_start
        self.NOISE_END = noise_end
        self.NOISE_DECAY = noise_decay
        self.EPISODES_ROLLOUT = EPISODES_ROLLOUT


class DDPG:

    def __init__(self,config):
        self.config = config
        self.state_size = config.state_size
        self.action_size = config.action_size


        self.actor_local = Actor(self.state_size,self.action_size,2).to(device)
        self.actor_target = Actor(self.state_size, self.action_size,2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = config.LR_ACTOR)

        self.critic_local = Critic(self.state_size* config.n_agents, self.action_size * config.n_agents,2).to(device)
        self.critic_target = Critic(self.state_size  * config.n_agents, self.action_size * config.n_agents,2).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = config.LR_CRITIC,)

        self.t_step = 0
        
        self.soft_update(self.critic_local, self.critic_target,1)
        self.soft_update(self.actor_local, self.actor_target,1)

        self.noise_factor = config.NOISE_START



    def act(self, state,episode, add_noise=True):
        """Returns actions for given state as per current policy."""

        if episode > self.config.EPISODES_ROLLOUT:
            self.noise_factor = max(self.config.NOISE_END, self.config.NOISE_DECAY ** (episode - self.config.EPISODES_ROLLOUT))


        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()        
        
        action += self.noise_factor * self.add_noise2()
        return np.clip(action, -1, 1)

    
    def learn(self, experiences, gamma):
        full_states, actor_full_actions, full_actions, agent_rewards, agent_dones, full_next_states, critic_full_next_actions = experiences
        
        Q_targets_next = self.critic_target(full_next_states, critic_full_next_actions)
        Q_targets = agent_rewards + (self.config.GAMMA * Q_targets_next * (1 - agent_dones))
        Q_expected = self.critic_local(full_states, full_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic_local(full_states, actor_full_actions).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def soft_update_all(self):
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target,self.config.TAU)
        self.soft_update(self.actor_local, self.actor_target,self.config.TAU) 

    
    def add_noise2(self):
        noise = 0.5*np.random.randn(1,self.action_size) #sigma of 0.5 as sigma of 1 will have alot of actions just clipped
        return noise  
        
                   

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data) 


