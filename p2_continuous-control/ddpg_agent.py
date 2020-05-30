import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import random

from models import Actor, Critic
from noise import OUNoise
from replay import ReplayBuffer


BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99 
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
UPDATE_EVERY = 10
EPOCH = 10

#WEIGHT_DECAY = 0.0001

device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu")


class DDPG:

    def __init__(self,state_size ,action_size,random_seed):
        self.state_size = state_size
        self.action_size = action_size


        self.actor_local = Actor(state_size,actor_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.paramters(), lr = LR_ACTOR)

        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC)

        self.soft_update(self.critic_local,self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

        self.noise = OUNoise(action_size,random_seed)
        self.memory = ReplayBuffer(random_seed)

        self.t_step = 0

    def step(self, states,actions,rewards,next_states,dones):

        for state,action,reward,next_state,done in zip(states,actions,rewards, next_states,dones):
            self.memory.add(state,action,reward,next_state,done)

        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY


        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                for _ in range(EPOCH):
                    experiences = self.memory.sample()
                    self.learn(experiences)

    
    def act(self, state ,add_noise = False):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        actions += self.noise.sample()

        return np.clip(actions , -1 ,1)

    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        # get actor from the actor target network(Deterministic) --> Using improvements from DDQN
        Q_targets_next = self.critic_target(next_states,actions_next)
        # get single value representing the action value for action actions_next
        Q_targets = rewards + gamma *(Q_targets_next * (1-dones))


        Q_expected = self.critic_local(states,actions)
        critic_loss = F.mse_loss(Q_expected,Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()


        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states,actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data) 



BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 3e-3        # learning rate of the critic
WEIGHT_DECAY = 0
UPDATE_EVERY = 20
EPSILON = 1.0           # explore->exploit noise process added to act step
EPSILON_DECAY = 1e-6    # decay rate for noise process
EPOCH = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DDPG:

    def __init__(self,state_size ,action_size,random_seed,n_agents):
        self.state_size = state_size
        self.action_size = action_size


        self.actor_local = Actor(state_size,action_size,2).to(device)
        self.actor_target = Actor(state_size, action_size,2).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR)

        self.critic_local = Critic(state_size, action_size,2).to(device)
        self.critic_target = Critic(state_size, action_size,2).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC,)

        self.memory = ReplayBuffer(random_seed,BUFFER_SIZE)
        self.noise = OUNoise((n_agents, action_size), random_seed)
        
        
        self.soft_update(self.critic_local, self.critic_target,TAU)
        self.soft_update(self.actor_local, self.actor_target,TAU)   
        
        self.epsilon = EPSILON
        self.t_step = 0

    def step(self, states,actions,rewards,next_states,dones):

        for state,action,reward,next_state,done in zip(states,actions,rewards, next_states,dones):
            self.memory.add(state,action,reward,next_state,done)

        
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            if len(self.memory) > BATCH_SIZE:
                for _ in range(EPOCH):                    
                    experiences = self.memory.sample(BATCH_SIZE)
                    self.learn(experiences)

    def reset(self):
        self.noise.reset()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()        
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)



    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        
        Q_targets_next = self.critic_target(next_states, self.actor_target(next_states))
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic_local(states, self.actor_local(states)).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target,TAU)
        self.soft_update(self.actor_local, self.actor_target,TAU)   
        
                   

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data) 





