import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import random
 

SGD_EPOCH = 10              
GAMMA = 0.99 
gradient_clip= 5
BETAS = (0,0)
LR = 3e-4        
BATCH_SIZE = 128

class Memory:
    def __init__(self):
        self.initialize_zeros()
        
    def initialize_zeros(self,MAX_T = 1000 ,num_agent = 20 , action_size = 4, state_size = 33):
        self.actions = np.zeros((MAX_T, num_agent ,action_size))
        self.states = np.zeros((MAX_T, num_agent ,state_size))
        self.logprobs = np.zeros((MAX_T, num_agent))
        self.rewards = np.zeros((MAX_T, num_agent))
        self.is_terminals = np.zeros((MAX_T, num_agent))
    def clear_memory(self):
        
        self.initialize_zeros()
        

class PPO:
    def __init__(self, state_size , action_size):
        self.policy = ActorCritic(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr = LR, betas = BETAS)
        self.policy_old = ActorCritic(state_size, action_size)
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = Memory()
        self.step_t = 0

        self.MseLoss = nn.MSELoss()


    def act(self, state, t):
        state = torch.FloatTensor(state.reshape(20,-1)).to(device)
        return self.policy_old.act(state,self.memory,t).cpu().data.numpy().flatten()


    def collect_trajectories(self,rewards,dones,t):
        
        '''
        After each step we will have, for each agent
        state : [33]
        action : [4]
        l_prob : [1]
        reward : [1]
        is_terminal : [1]
        
        for 20 agents,
        state : [20,33]
        action : [20,4]
        l_prob : [20,1]
        reward : [20,1]
        is_terminal : [20,1]
        
        --> next find_advantage. 
        
        '''
        
        count = 0
        for idx ,(reward,done) in enumerate(zip(rewards,dones)):
            self.memory.rewards[t,idx] = reward#[1]
            self.memory.is_terminals[t,idx] = done #[1]
              
        
        
    def find_advantage(self,states,rewards,dones):
        
        '''
        Find advantage for each state,action,reward tuple
        
        The entire memory for a horizon of length T will be:
        
        state : [T,20,33]
        action : [T,20,4]
        l_prob : [T,20,1]
        reward : [T,20,1]
        is_terminal : [T,20,1]
        '''
        
        T = states.size(0)
        advantages = np.zeros((T,20))
        returns = np.zeros((T,20))

        
        for idx in range(20):
            ret = self.policy_old.find_state_value(states[T-1,idx,:]).squeeze(-1).detach()
            returns[T-1,idx] = ret
            for i in reversed(range(T-1)):
                ret = rewards[i,idx] + GAMMA * (1 - dones[i,idx]) * ret
                state = states[i,idx,:] # T,33
                state_value = self.policy_old.find_state_value(state).squeeze(-1).detach() # T,
                advantages[i,idx] = ret - state_value
                returns[i,idx] = ret
                
        return torch.from_numpy(advantages).to(device),torch.from_numpy(returns).to(device)
            

    def step(self,epsilon,beta):
        self.learn(epsilon ,beta)
        self.memory.clear_memory()
        

    def learn(self, epsilon, beta):
        
        states = torch.from_numpy(self.memory.states).float().to(device).detach()
        actions = torch.from_numpy(self.memory.actions).float().to(device).detach()
        rewards = torch.from_numpy(self.memory.rewards).float().to(device).detach()    
        is_terminals = torch.from_numpy(self.memory.is_terminals.astype(np.uint8)).float().to(device).detach()
        old_probs = torch.from_numpy(self.memory.logprobs).float().to(device).detach()            
            
        advantages,returns = self.find_advantage(states,rewards,is_terminals)
        
        advantages = (advantages - advantages.mean())/advantages.std()
            
            
        
        for _ in range(SGD_EPOCH):
            sampler = random_sample(np.arange(states.size(0)), BATCH_SIZE)
            for batch_indices in sampler:
                batch_indices = torch.tensor(batch_indices).long()
                sampled_states = states[batch_indices] # BATCH * 33
                sampled_actions = actions[batch_indices] # BATCH * 4
                sampled_log_probs_old = old_probs[batch_indices] # BATCH,
                sampled_advantages = advantages[batch_indices] # BATCH,
                sampled_returns = returns[batch_indices]
                
                logprobs, state_values, dist_entropy = self.policy.evaluate(sampled_states,sampled_actions)
                
                ratios = torch.exp(logprobs - sampled_log_probs_old)
                
            
                surr1 = sampled_advantages * ratios
                surr2 = torch.clamp(ratios,1 - epsilon , 1+ epsilon) * sampled_advantages 
                                
                loss1 =  torch.min(surr1,surr2)
                loss2 = self.MseLoss(sampled_returns , state_values)
                loss3 = beta * dist_entropy
            
                
                loss = - loss1.mean() + loss2 - loss3.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
