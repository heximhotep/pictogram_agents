#!/usr/bin/env python3
import sys
sys.path.append('/home/studio/Documents/aman/pictogram_agents')
sys.path.append('/home/studio/Documents/aman/pictogram_agents/robosumo/')

import robosumo.envs

import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
import starfish

from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import MlpPolicy, MlpLstmPolicy
import gym
import tensorflow as tf
import numpy as np
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

import time

class MonitorTuple(bench.Monitor):

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        info = {'agent ' + str(i): subinfo for i,subinfo in enumerate(info)}
        self.rewards.append(max(rew))
        if any(done):
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(epinfo)
                self.f.flush()
            info['episode'] = epinfo
        self.total_steps += 1
        return (ob, rew, done, info)
    

class DummyVecEnvTuple(DummyVecEnv):
    def __init__(self, env_fns):
        super(DummyVecEnvTuple, self).__init__(env_fns)

        obs_spaces = self.observation_space.spaces if isinstance(self.observation_space, gym.spaces.Tuple) else (self.observation_space,)

        self.nagents = len(self.observation_space.spaces)
        self.buf_dones = np.zeros((self.num_envs * self.nagents,), dtype=np.bool)
        self.buf_rews  = np.zeros((self.num_envs * self.nagents,), dtype=np.float32)
        self.actions = None

    def step_wait(self):
        for i in range(self.num_envs):
            obs_tuple, reward, done, info = self.envs[i].step(self.actions[i])
            for j in range(self.nagents):
                idx = i * self.nagents + j
                self.buf_rews[idx] = reward[j]
                self.buf_dones[idx] = done[j]
                if self.buf_dones[idx]:
                    obs_tuple = self.envs[i].reset()
                    for k in range(self.nagents):
                        self.buf_dones[i * self.nagents + k] = True
            self.buf_infos[i] = info
           
            if isinstance(obs_tuple, (tuple, list)):
                for t,x in enumerate(obs_tuple):
                    self.buf_obs[t][i] = x
            else:
                self.buf_obs[0][i] = obs_tuple
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones),
                self.buf_infos.copy())


class VecNormalizeTuple(VecNormalize):
    """
    Vectorized environment base class
    """

    def __init__(self, venv):
        super().__init__(venv)
        self.nagent = len(self.observation_space.spaces)
        self.ret = np.zeros(self.num_envs * self.nagent)
    
    def _concat_observations(self, obs):
        return np.concatenate([np.stack(sub_obs, axis=0) for sub_obs in obs], axis=0)

    def _unconcat_observations(self, obs, sub_ob_dim, batch_dim):
        obs = obs.reshape(sub_ob_dim, batch_dim, obs.shape[-1])
        ret = tuple(tuple(sub_ob for sub_ob in sub_obs) for sub_obs in obs)
        return ret


    def _obfilt(self, obs):
        needs_un_concat = False
        if isinstance(obs, (tuple, list)):
            needs_un_concat = True
            sub_ob_dim = len(obs) 
            batch_dim = obs[0].shape[0]
            obs = self._concat_observations(obs)
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            
            if needs_un_concat:
                obs = self._unconcat_observations(obs, sub_ob_dim, batch_dim)
            return obs
        else:
            if needs_un_concat:
                obs = self._unconcat_observations(obs, sub_ob_dim, batch_dim)

            return obs

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        return self._obfilt(obs)

def train(env_id, num_timesteps, seed, load_path=False):
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    is_sumo = 'sumo' in env_id.lower()

    def make_env():
        env = gym.make(env_id)
        if is_sumo:
            env = MonitorTuple(env, logger.get_dir(), allow_early_resets=True)
        else:
            env = bench.Monitor(env, logger.get_dir())
        return env
    

    env = DummyVecEnvTuple([make_env] * 12)
    env = VecNormalizeTuple(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        save_interval=20,
        load_path=load_path,
        sumo=is_sumo)


def main():
    parser = mujoco_arg_parser()
    parser.add_argument('--logdir')
    parser.add_argument('--load-path', default=None)
    args = parser.parse_args()
    logger.configure(dir=args.logdir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, load_path=args.load_path)


if __name__ == '__main__':
    main()
