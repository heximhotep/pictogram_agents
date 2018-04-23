import numpy as np
import random

train_envs = ['findtarget']
test_envs = ['findtarget']

def generate_data_action(env):
     a = env.action_space.sample()
     return a


def adjust_obs(obs):
    return obs.astype('float32') / 255.

