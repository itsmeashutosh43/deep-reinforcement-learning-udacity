import random 
import numpy as np 
import torch 
from collections import namedtuple, deque

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

        full_states = torch.from_numpy(np.array([e.full_state for e in experiences if e is not None])).float().to(DEVICE)
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        full_next_states = torch.from_numpy(np.array([e.full_next_state for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return (full_states, states, actions, rewards, full_next_states, next_states, dones)

    def __len__(self):
        return len(self.memory)
