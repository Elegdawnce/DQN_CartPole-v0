import torch
import torch.nn as nn
import numpy as np
import gym

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9 # greedy policy
GAMMA = 0.9 # reward discount
TARGET_REPLACE_ITER = 100 # target update frequency
MEMORY_CAPACITY = 100
WEIGHT_DECAY = 0.0001
LR_DECAY_STEP_SIZE = 6


env = gym.make('CartPole-v0')  # 创建一个实验场所，立杆实验
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

print(N_STATES)