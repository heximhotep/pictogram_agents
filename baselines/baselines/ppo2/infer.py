#!/usr/bin/env python3

import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
import dm_control.suite
import gym
import starfish


def infer(env_id, load_path, num_timesteps, seed, save_video):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)

        return env
    #env = DummyVecEnv([make_env])
    #env = VecNormalize(env)
    env = make_env()
    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.run(policy=policy, load_path=load_path, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        save_interval=10000)


def main():
    parser =  mujoco_arg_parser()
    parser.add_argument('--load-path')
    parser.add_argument('--save-video')
    args = parser.parse_args()
    logger.configure()
    infer(args.env, load_path=args.load_path, 
          num_timesteps=args.num_timesteps, 
          seed=args.seed,
          save_video=args.save_video)

if __name__ == '__main__':
    main()