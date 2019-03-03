import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import torch.optim as optim

import dataset
import model


def train():
    # 初始化网络
    dqn = model.DQN()
    print(dqn)

    # # 获取计算设备
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    #     num_gpu = torch.cuda.device_count()
    #     if num_gpu > 1:
    #         dqn = nn.DataParallel(dqn)
    #     print('Using %d GPU...' % num_gpu)
    # else:
    #     device = torch.device('cpu')
    #     print('Using CPU...')
    # # 网络转移到设备上
    # dqn.to(device)

    print('\nCollecting experience...')
    for epoch in range(400):
        s0 = dataset.env.reset()
        # s0 = s0.to(device)
        ep_r = 0
        while True:
            dataset.env.render()
            a = dqn.choose_action(s0)
            # take action
            s1,r,done,info = dataset.env.step(a)
            # modify the reward 因为reward一直是1
            x,x_dot,theta,theta_dot = s1
            r1 = (dataset.env.x_threshold - abs(x)) / dataset.env.x_threshold - 0.8
            r2 = (dataset.env.theta_threshold_radians - abs(theta)) / dataset.env.theta_threshold_radians - 0.5
            r = r1+ r2

            # s0,a,r,s1 = s0.to(device),a.to(device),r.to(device),s1.to(device)

            dqn.store_transition(s0,a,r,s1)

            ep_r += r
            if dqn.memory_counter > dataset.MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep:',epoch,'\tEp_r:',round(ep_r,2))

            if done:
                break
            s0 = s1







if __name__ == '__main__':
    train()