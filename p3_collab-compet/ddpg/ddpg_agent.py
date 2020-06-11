import torch
import torch.nn as nn 
import torch.nn.optim as optim
import torch.nn.functional as F 
import numpy as np 
import random

from models import Actor, Critic
from noise import OUNoise
from replay import ReplayBuffer


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        self.memory = ReplayBuffer(config.random_seed,config.BUFFER_SIZE)
        self.noise = OUNoise(self.action_size, config.random_seed) 
        
        self.t_step = 0
        
        self.soft_update(self.critic_local, self.critic_target,1)
        self.soft_update(self.actor_local, self.actor_target,1)

        self.noise_factor = config.NOISE_START


    def reset(self):
        self.noise.reset()

    def act(self, state,episode, add_noise=True):
        """Returns actions for given state as per current policy."""

        if episode > self.config.EPISODES_ROLLOUT:
            self.noise_factor = max(self.config.NOISE_END, self.config.NOISE_DECAY ** (episode - EPISODES_ROLLOUT))


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





