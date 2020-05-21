import numpy as np
import random
from collections import namedtuple,deque

from models import Network


import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4   

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    
    def __init__(self,state_size , action_size , seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.Q = Network(self.state_size , self.action_size , self.seed)
        self.Q_dash = Network(self.state_size,self.action_size , self.seed)
        
        self.optimizer = optim.Adam(self.Q.parameters(), lr = LR)
        
        self.replay = ReplayBuffer(self.seed)
        self.t_step = 0
        
    def step(self, state , action , reward ,next_state, done):
        self.replay.add(state , action , reward ,next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        if self.t_step == 0:
            if len(self.replay) > BATCH_SIZE:
                experiences = self.replay.sample()
                self.learn_dqn(experiences,GAMMA)
                
                
    def act(self, state, eps=0.):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.Q.eval()
        with torch.no_grad():
            # done to avoid bt
            action_values = self.Q(state)
        self.Q.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
        
    def learn_dqn(self, experiences, gamma):
        '''
        Simple dqn with fixed target Q' and experience replay
        '''
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.q_dash(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        # only get reward if its done
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.q(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.q, self.q_dash, TAU)

        
    def learn_ddqn(self, experiences, gamma):
        #double deep q learning
        
        states, actions, rewards, next_states, dones = experiences
        
        best_action_arg = self.q(next_states).detach()
        a_best = best_action_arg.max(1)[1]
        Q_targets_next = self.q_dash(next_states).detach().gather(1,a_best.unsqueeze(1))
        #Q_targets_next = Q_targets_all[np.arange(BATCH_SIZE), a_best].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.q(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

                
                
                
        
    
        
        
class ReplayBuffer():
    
    def __init__(self, seed):
        
        self.memory = deque(maxlen = BUFFER_SIZE)
        self.experience = namedtuple("Experience", field_names = ["state","action","reward","next_state","done"])
        self.seed = random.seed(seed)
        
        
    def add(self, state,action,reward, next_state, done):
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        masks = random.sample(self.memory, BATCH_SIZE)
        
        states = torch.from_numpy(np.vstack([e.state for e in masks if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in masks if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in masks if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in masks if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in masks if e is not None])).float().to(device)
        
        return (states,actions,rewards, next_states, dones)
    
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
        
        
    