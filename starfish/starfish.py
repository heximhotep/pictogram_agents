import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
class SwimX(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, os.path.dirname(__file__) + '/assets/starfish.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(self.sim.data.qacc[:]).sum()
        reward = reward_fwd + reward_ctrl
        if np.isnan(reward):
            print("NANM")
        if np.isnan(self.sim.data.qacc).any():
            print("NNNN")
        ob = self._get_obs()

        if xposafter > 20:
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
