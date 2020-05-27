import torch
import torch.nn as nn 
import torch.nn.Functional as F 
import numpy as np 
import random
import copy
from collections import namedtuple,deque


from models import Actor, Critic

BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128
GAMMA = 0.99 
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0.0001

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
        self.noise = OUNoise(action_size,random_seed)

        self.memory = ReplayBuffer(random_seed)

    def step(self, state,action,reward,next_state,done):
        self.memory.add(state,action,reward,next_state,done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences,GAMMA)

    def act(self, state ,add_noise = False):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()

        return np.clip(action , -1 ,1)

    
    def learn(self.experiences):
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(states)
        # get actor from the actor target network(Deterministic) --> Using improvements from DDQN
        Q_targets_next = self.critic_target(states,actions_next)
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




class OUNoise:

    def __init__(self,action_size,random_seed , mu = 0, theta = 0.15 , sigma  =0.2):
        self.seed = random.seed(random_seed)
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        #x = self.update
        pass




class ReplayBuffer:

    def __init__(self,seed = 0):
        self.memory = deque(maxlen = BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names = ["state","action","reward","next_state","done"])

        self.seed = random.seed(seed)

    def add(self, state,action,reward, next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)


    def sample(self):
        experiences = random.sample(self.memory, k=BATCH_SIZE)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
