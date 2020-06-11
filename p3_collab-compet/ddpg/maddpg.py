from ddpg_agent import DDPG,Config
from replay import ReplayBuffer

class MADDPG:
    def __init__(self,state_size,action_size,num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.whole_action_dim = self.action_size * self.num_agents
        self.memory = ReplayBuffer(seed = 10, buffer_size = BUFFER_SIZE)

        self.maddpg_agents = [DDPG(Config(state_size = state_size,action_size = action_size,random_seed = 2,n_agents = num_agents)) for _ in range(num_agents)]

        self.episodes_rollout = EPISODES_ROLLOUT 

    def reset(self):
        for agent in self.maddpg_agents:
            agent.reset()

    def step(self,i_episodes,states,actions,rewards,next_states,dones):
        '''
        states.shape = 2,24
        actions.shape = 2.2

        '''
        states1d = states.reshape(-1) # 2*24
        next_states1d = next_states.reshape(-1)
        self.memory.add(states1d, states,actions,rewards,next_states1d,next_states,dones)


        if len(self.memory) > BATCH_SIZE and i_episodes > self.episodes_rollout:
            for _ in range(EPOCHS):
                for an in range(self.num_agents):
                    samples = self.memory.sample(BATCH_SIZE)
                    self.learn(samples,an,GAMMA)

                self.soft_updates_all()

    def soft_updates_all(self):
        for agent in self.maddpg_agents:
            agent.soft_update_all()


    def learn(self, samples, an , gamma):
        '''
        shapes of variables
        full_states = batchsize , 2*24
        states = batchsize, 2, 24
        '''
        full_states, states,actions, rewards, full_next_states,next_states, dones = samples

        critic_full_next_actions = torch.zeros(states.shape[:2] + (self.action_size, ), device = device, dtype = torch.float)

        for a_id, agent in enumerate(self.maddpg_agents):
            next_state = next_states[:,a_id,:]
            critic_full_next_actions[:,a_id,:] = agent.actor_target.forward(next_state)


        critic_full_next_actions = critic_full_next_actions.view(-1, self.whole_action_dim)
        agent = self.maddpg_agents[an]
        agent_state = states[:,an,:]
        actor_full_actions = actions.clone()
        actor_full_actions[:,an,:] = agent.actor_local.forward(agent_state)
        actor_full_actions = actor_full_actions.view(-1,self.whole_action_dim)   

        full_actions = actions.view(-1, self.whole_action_dim)


        agent_rewards = rewards[:,an].view(-1,1) 
        agent_dones = dones[:,an].view(-1,1) 
        experiences = (full_states, actor_full_actions, full_actions, agent_rewards, \
                       agent_dones, full_next_states, critic_full_next_actions)
        agent.learn(experiences, gamma)


    def act(self, full_states, i_episode, add_noise=True):
        # all actions between -1 and 1
        actions = []
        for agent_id, agent in enumerate(self.maddpg_agents):
            action = agent.act(np.reshape(full_states[agent_id,:], newshape=(1,-1)), i_episode, add_noise)
            action = np.reshape(action, newshape=(1,-1))            
            actions.append(action)
        actions = np.concatenate(actions, axis=0)
        return actions