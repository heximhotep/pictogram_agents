import numpy as np
#import gym
import gym
import dm_control2gym

def make_env(env_name, seed=-1, render_mode=False):
  if env_name == 'car_racing':
    env = CarRacing()
    if (seed >= 0):
      env.seed(seed)
  
  if env_name == 'findtarget':
  	env = dm_control2gym.make(domain_name='findtarget', task_name='walk')
  else:
    print("couldn't find this env")


  return env
