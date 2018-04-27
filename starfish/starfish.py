import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.robotics import robot_env
from mujoco_py import MujocoException
from collections import OrderedDict
import os

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class SwimX(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, large=False):
        model = os.path.dirname(__file__) + '/assets/starfish.xml'
        if large:
            model = os.path.dirname(__file__) + '/assets/starfish_novis.xml'
        mujoco_env.MujocoEnv.__init__(self, model, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.get_body_com('agent0/root')[0]
        try:
            self.do_simulation(a, self.frame_skip)
        except MujocoException as e:
            print("err:", e)
            ob = self.reset_model()
            return ob, -1, True, dict(reward_fwd=0, reward_ctrl=-1)

        xposafter = self.get_body_com('agent0/root')[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(self.sim.data.qacc[:]).sum()
        reward = reward_fwd + reward_ctrl

        ob = self._get_obs()

        if np.linalg.norm(self.get_body_com('agent0/root')) > 10:
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

    def step(self, action):
        to_vec = self.get_body_com('agent0/root') - self.get_body_com('target')

        dist = np.linalg.norm(to_vec)
        reward_dist = -dist
        
        if dist > 6:
            term = True
        else:
            term = False

        if dist < 0.1:
            term = True

        # this is needed bc before the initial state is set the MujocoEnv class tries to run this
        if dist < 0.0001:
            reward_dist = -4
            term = False
            
        energy_penalty = -np.linalg.norm(action) * 0.0001
        reward = reward_dist + energy_penalty
        try:
            self.do_simulation(action, self.frame_skip)
        except MujocoException as e:
            print("err:", e)
            ob = self.reset_model()
            return ob, -100, True, dict(reward_dist=0, reward_ctrl=-1)

        observation = self._get_obs()
        return observation, reward, term, dict(reward_dist=reward_dist, 
                                               energy_penalty=energy_penalty)

    def _get_obs(self):
        dist_to_target = self.get_body_com('agent0/root') - self.get_body_com('target')
        target_pos = self.get_body_com('target')
        joint_vel = self.sim.data.qvel[:]
        rel_pos = self.sim.data.body_xpos[:-1] - self.sim.data.subtree_com[:-1]
        return np.concatenate((dist_to_target, target_pos, joint_vel, rel_pos.flat[:]))
        

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
    
    
class FindTargetHER(robot_env.RobotEnv):
    ''' same as FindTarget except as as a RobotEnv  '''
    
    def __init__(self, reward_type='sparse', distance_threshold=0.09):
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        super(FindTargetHER, self).__init__(
            model_path=os.path.dirname(__file__) + '/assets/starfish_novis.xml',
            n_substeps=20, 
            initial_qpos={}, 
            n_actions=24)

    def compute_reward(self, achieved_goal, goal, info):
        dist = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(dist > self.distance_threshold).astype(np.float32)
        else:
            return -dist

    def _set_action(self, action):
        self.sim.data.ctrl[:] = action 

    def _body_pos_by_name(self, name):
        return self.sim.data.body_xpos[self.sim.model.body_name2id(name)]

    def _get_obs(self):
        self_pos = self._body_pos_by_name('agent0/root')
        target_pos = self._body_pos_by_name('target')
        dist_to_target = self_pos - target_pos

        joint_vel = self.sim.data.qvel[:]
        joint_pos = self.sim.data.qpos.flat[:]
        
        obs = OrderedDict({
            'observation': np.concatenate((dist_to_target, target_pos, joint_vel, joint_pos)),
            'desired_goal': target_pos,
            'achieved_goal': self_pos
        })
        
        return obs
        
    def _is_success(self, achieved_goal, desired_goal):
        dist = goal_distance(achieved_goal, desired_goal)
        return (dist < self.distance_threshold).astype(np.float32)


    def _sample_goal(self):
        init_torso = self._body_pos_by_name('agent0/root')
        radius = 1.5
        goal = init_torso + self.np_random.uniform(-radius, radius, size=3)
        target_id = self.sim.model.body_name2id('target')
        self.sim.data.body_xpos[target_id] = goal.copy()
        return goal.copy()
