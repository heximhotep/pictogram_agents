#!/usr/bin/env python3

import argparse
from baselines.common.cmd_util import mujoco_arg_parser
from baselines import bench, logger
import dm_control.suite
import gym
import starfish


def train(env_id, num_timesteps, seed):
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
        if env_id != "starfish":
            env = gym.make(env_id)
        else:
            env = dm_control2gym.make(domain_name='findtarget', task_name='starfish_walk')
            pdb.set_trace()
            #env._max_episode_steps = 256

        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env] * 12)
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    ppo2.learn(policy=policy, env=env, nsteps=2048, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        save_interval=20)


def main():
    parser = mujoco_arg_parser()
    parser.add_argument('--logdir')
    args = parser.parse_args()
    logger.configure(dir=args.logdir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
