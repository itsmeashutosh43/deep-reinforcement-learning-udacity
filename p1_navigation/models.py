import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    '''
    Class to map all states to actions
    '''
    
    def __init__(self, stateSize , actionSize , seed , fc1 = 128 , fc2 = 128, fc3 = 64):
        super(Network,self).__init__()
                
        self.fc1 = nn.Linear(stateSize , fc1)
        self.fc2 = nn.Linear(fc1 , fc2)
        self.fc3 = nn.Linear(fc2 , fc3)
        self.fc4 = nn.Linear(fc3, actionSize)
        
    def forward(self,state):
        action = F.relu(self.fc1(state))
        action = F.relu(self.fc2(action))
        action = F.relu(self.fc3(action))
        action = self.fc4(action)
        
        return action
        
    