import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import torch.optim as optim

import dataset

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc = nn.Linear(dataset.N_STATES,50)
        self.fc.weight.data.normal_(0,0.1)
        self.relu = nn.ReLU()
        self.out = nn.Linear(50,dataset.N_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)

        return x


class DQN(object):
    def __init__(self):
        self.qeval_net = Net()
        self.qtarget_net = Net()

        self.learn_step_counter = 0 # for target updating
        self.memory_counter = 0 # for storing memory
        self.memory = np.zeros((dataset.MEMORY_CAPACITY, dataset.N_STATES*2+2)) # initialize memory
        self.optimizer = optim.Adam(self.qeval_net.parameters(),lr=dataset.LR,weight_decay=dataset.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, dataset.LR_DECAY_STEP_SIZE)
        self.criterion = nn.MSELoss()

    def choose_action(self,x):
        # 增加第一维
        x = torch.unsqueeze(torch.FloatTensor(x),0)
        # 随机选取动作的概率（90%选取greedy）
        if np.random.uniform() < dataset.EPSILON: # greedy
            actions_value = self.qeval_net.forward(x)
            action = torch.max(actions_value,1)[1].numpy()
            if dataset.ENV_A_SHAPE == 0:
                action = action[0]
            else:
                action.reshape(dataset.ENV_A_SHAPE)
        else: # random
            action = np.random.randint(0,dataset.N_ACTIONS)
            if dataset.ENV_A_SHAPE == 0:
                # action = action[0]
                action = action
            else:
                action.reshape(dataset.ENV_A_SHAPE)
        return action

    def store_transition(self,s_0,a,r,s_1):
        transition = np.hstack((s_0,[a,r],s_1))
        index = self.memory_counter % dataset.MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.memory_counter += 1

    def learn(self):
        self.scheduler.step()
        # target parameter update
        if self.learn_step_counter % dataset.TARGET_REPLACE_ITER == 0:
            self.qtarget_net.load_state_dict(self.qeval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(dataset.MEMORY_CAPACITY,dataset.BATCH_SIZE)
        b_memory = self.memory[sample_index,:]
        b_s0 = torch.FloatTensor(b_memory[:,:dataset.N_STATES])
        b_a = torch.LongTensor(b_memory[:,dataset.N_STATES:dataset.N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:,dataset.N_STATES+1:dataset.N_STATES+2])
        b_s1 = torch.FloatTensor(b_memory[:,-dataset.N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.qeval_net(b_s0).gather(1,b_a) # shape (batch,1)
        q_next = self.qtarget_net(b_s1).detach() # detach from graph, don't backpropagate
        q_target = b_r + dataset.GAMMA * q_next.max(1)[0].view(dataset.BATCH_SIZE,1) # shape (batch,1)
        loss = self.criterion(q_eval,q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()