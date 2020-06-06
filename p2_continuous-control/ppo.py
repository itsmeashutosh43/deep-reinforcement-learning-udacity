from models import ActorCritic
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import random
 
device = torch.device("cuda:0" if torch.cuda.is_available() else  "cpu")

ACTION_STD = 0.5            # constant std for action distribution (Multivariate Normal)
SGD_EPOCH = 8               
GAMMA = 0.99                # discount factor    
LR = 0.0003                 # parameters for Adam optimizer
BETAS = (0.9, 0.999)
UPDATE_EVERY = 20

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self, state_size , action_size):
        self.random_seed = self.cofig.random_seed
        self.policy = ActorCritic(state_size, action_size, ACTION_STD, device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = LR, betas = betas)
        self.policy_old = ActorCritic(state_size, action_size, ACTION_STD, device)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = Memory()
        self.step_t = 0

        self.MseLoss = nn.MSELoss()


    def act(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        return self.policy_old.act(state,self.memory).cpu().data.numpy().flatten()


    def step(self,reward,done,epsilon,beta):
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)

        self.step_t = (self.step_t + 1) % UPDATE_EVERY

        if self.step_t == 0:
            self.learn(epsilon ,beta)

    def learn(self, epsilon, beta):
        discount = GAMMA**np.arange(len(self.memory.rewards))
        rewards = np.asarray(self.memory.rewards) * discount[:,np.newaxis]
        rewards_future = rewards[::-1].cumsum(axis = 0)[::-1]

        states = torch.squeeze(torch.stack(self.memory.states).to(device),1).detach()
        actions = torch.squeeze(torch.stack(self.memory.actions).to(device),1).detach()
        old_probs = torch.squeeze(torch.stack(self.memory.logprobs).to(device),1).detach()

        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]


        for _ in range(SGD_EPOCH):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states,actions)

            ratios = torch.exp(logprobs - old_probs.detach())

            advantages = rewards_normalized - state_values.detach()
            surr1 = advantages * ratios
            surr2 = torch.clamp(ratios,1 - epsilon , 1+ epsilon) * advantages 

            loss = - torch.min(surr1,surr2) + 0.5 * self.MseLoss(state_values, rewards) - beta * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


        self.policy_old.load_state_dict(self.policy.state_dict())

        

