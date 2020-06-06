import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions import MultivariateNormal
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

class ActorCritic(nn.Module):

    def __init__(self,state_size,action_size):
        super(ActorCritic,self).__init__()

        

        self.actor = nn.Sequential(
            nn.Linear(state_size,256),
            nn.ReLU(inplace = True),
            nn.Linear(256,128),
            nn.ReLU(inplace = True),
            nn.Linear(128,action_size),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_size,256),
            nn.ReLU(inplace = True),
            nn.Linear(256,128),
            nn.ReLU(inplace = True),
            nn.Linear(128,1)
        )
        
        self.action_size = action_size
        
        self.std = nn.Parameter(torch.zeros(action_size))

        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)


    def forward(self):
        raise NotImplementedError

    def act(self,states,memory,t):
                
        action_mean = self.actor(states)
            
        dist = torch.distributions.Normal(action_mean, F.softplus(self.std))
        actions = dist.sample() # --> 20 ,4
        log_prob = dist.log_prob(actions).sum(dim = -1).unsqueeze(-1) #--> 20, 1
        log_prob = torch.sum(log_prob, dim=-1) # --> 20,1
        
        for idx,(state,action,l_prob) in enumerate(zip(states,actions,log_prob)):
            memory.states[t,idx,:] = state # [33]
            memory.actions[t,idx,:] = action# [4] 
            memory.logprobs[t,idx] = l_prob# [1]
        
        return actions.detach()

    def evaluate(self, state ,action):
        
        action_mean = self.actor(state)
        dist = torch.distributions.Normal(action_mean, F.softplus(self.std))
        log_prob = dist.log_prob(action).sum(dim = -1) # [1]
        entropy = dist.entropy().sum(-1) # [1]
        return log_prob, self.critic(state).squeeze(-1), entropy
    
    def find_state_value(self,state):
        return self.critic(state)

