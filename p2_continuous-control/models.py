import torch
import torch.nn as nn
import torch.nn.Functional as F 
import numpy as np

def init_helper( nn_size):
    in = nn_size.weight.data.size()[]
    offset = 1.0/np.sqrt(in)
    return (-offset, offset)


class Actor(nn.module):
    def __init__(self, state_dim, action_dim ,hidden1 = 64 , hidden2 = 64):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden2, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = nn.Tanh(fc2)

        return out 

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*init_helper(self.fc1))
        self.fc2.weight.data.uniform__(*init_helper(self.fc2))



class Critic(nn.module):
    def __init__ (self, state_dim, action_dim, hidden1 = 256, hidden2 = 128 , hidden3 = 128):
        super(Critic.self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1+ action_dim , hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3 , 1)

    def forward(self, state,action):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(torch.cat([out,action], dim = 1)))
        out = F.relu(self.fc3(out))
        return self.fc4(out)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*init_helper(self.fc1))
        self.fc2.weight.data.uniform__(*init_helper(self.fc2))
        self.fc3.weight.data.uniform_(*init_helper(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3,3e-3)