import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from mujoco_py import MujocoException
import os

class SwimX(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(__file__) + '/assets/starfish.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        try:
            self.do_simulation(a, self.frame_skip)
        except MujocoException as e:
            print("err:", e)
            ob = self.reset_model()
            return ob, -1, True, dict(reward_fwd=0, reward_ctrl=-1)

        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(self.sim.data.qvel[:]).sum()
        reward = reward_fwd + reward_ctrl
        if np.isnan(reward):
            print("NANM")
        if np.isnan(self.sim.data.qacc).any():
            print("NNNN")
        ob = self._get_obs()

        if np.linalg.norm(self.sim.data.qpos[:]) > 10:
            term = True
        else:
            term = False

        return ob, reward, term, dict(reward_fwd=reward_fwd, 
                                    reward_ctrl=reward_ctrl, r=reward_fwd, l=5.0)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return qvel.flat[:] # np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()


class FindTarget(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, frameskip=4, radius=4):
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(__file__) + '/assets/starfish.xml', frameskip)
        utils.EzPickle.__init__(self)
        self._initted = False
    def step(self, action):
        if not hasattr(self, '_initted'):
            to_vec = np.array([1, 1, 1])
        else:
            to_vec = self.get_body_com('agent0/root') - self.get_body_com('target')

        reward_dist = -np.linalg.norm(to_vec)
        energy_penalty = -np.linalg.norm(action)
        reward = reward_dist + energy_penalty
        try:
            self.do_simulation(action, self.frame_skip)
        except MujocoException as e:
            print("err:", e)
            ob = self.reset_model()
            return ob, -100, True, dict(reward_dist=0, reward_ctrl=-1)

        observation = self._get_obs()

        if np.linalg.norm(to_vec) > 6:
            term = True
        else:
            term = False

        if np.linalg.norm(to_vec) < 0.1:
            term = True

        return observation, reward, term, dict(reward_dist=reward_dist, 
                                               energy_penalty=energy_penalty)

    def _get_obs(self):
        dist_to_target = self.get_body_com('agent0/root') - self.get_body_com('target')
        target_pos = self.get_body_com('target')
        joint_vel = self.sim.data.qvel[:]
        return np.concatenate((dist_to_target, target_pos, joint_vel))
        

    def reset_model(self):
        self._initted = True
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        
        ground_pos = self.data.get_geom_xpos('ground')[:]
        max_iter = 1000
        while True:
            target_loc = self.np_random.uniform(low=-2.0, high=2.0, size=3)
            if target_loc[1] > ground_pos[1] or max_iter == 0:
                break
            max_iter -= 1

        self.model.body_pos[self.model.body_pos == self.get_body_com('target')] = target_loc
        
        return self._get_obs()