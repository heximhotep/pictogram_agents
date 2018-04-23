
# coding: utf-8

# In[8]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control, specs
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards

import random

import six
import numpy as np

from genome_synth import Genesis, get_model_and_assets


# In[11]:


_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .04
_WALK_SPEED = 1

SUITE = containers.TaggedTasks()


# In[ ]:


@SUITE.add('benchmarking')
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=random):
  """Returns the Walk task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  genesis = Genesis()
  genesis.from_xml('dust.xml')
  physics.set_genesis(genesis)
  task = FindTarget()
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP)


# In[9]:


class Physics(mujoco.Physics):
    def set_genesis(self, gen):
        self.gen = gen
    
    def upright(self):
        return self.named.data.xmat['root', 'zz']
    
    def root_vertical_orientation(self):
        """Returns the z-projection of the root orientation matrix."""
        return self.named.data.xmat['root', ['zx', 'zy', 'zz']]
    
    def joint_velocities(self):
        return self.named.data.qvel[:]
    
    def joint_angles(self):
        return self.data.qpos[7:] #skip the 7 DoFs of the root joint
    
    def extremities(self, rootname):
        jointnames = [joint for joint in self.gen.joints if rootname in joint]
        positions = []
        #could have some way of adding local frames here
        for jointname in jointnames:
            positions.append(self.geom_offset(rootname, jointname))
        
    def body_offset(self, _body1, _body2):
        data = self.named.data
        offset_global = data.xpos[_body2] - data.xpos[_body1]
        return offset_global.dot(data.xmat[_body1].reshape(3, 3))
    
    def geom_offset(self, _geom1, _geom2):
        #returns offset vector from _geom1 to _geom2 in _geom1's coordinate space
        data = self.named.data
        offset_global = data.geom_xpos[_geom2] - data.geom_xpos[_geom1]
        return offset_global.dot(data.geom_xmat[_geom1].reshape(3, 3))
    
    


# In[133]:


class FindTarget(base.Task):
    def __init__(self):
        super(FindTarget, self).__init__(random=None)
        
    def initialize_episode(self, physics):
        penetrating = True
        while penetrating:
          randomizers.randomize_limited_and_rotational_joints(physics, self.random)
          # Check for collisions.
          physics.after_reset()
          penetrating = physics.data.ncon > len(physics.gen.joints)
        targetpos = np.random.rand(3) * np.array([2, 2, 0.1])
        targetpos -= np.array([1, 1, 0])
        physics.named.data.xpos['target', ['x', 'y', 'z']] = targetpos
        
    def action_spec(self, physics):
        return specs.BoundedArraySpec([len(physics.gen.joints)], np.float32, minimum=-1., maximum=1.)

    def observation_spec(self, physics):
        an_obs = self.get_observation(physics)
        result = {}
        keys = six.iterkeys(an_obs)
        for key in keys:
            this_shape = an_obs[key].shape
            this_dtype = an_obs[key].dtype 
            result[key] = specs.ArraySpec(this_shape, this_dtype)
        return result

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['joint_angles'] = physics.joint_angles()
        obs['target_offset'] = physics.body_offset("root", "target")
        obs['vertical_orient'] = physics.root_vertical_orientation()
        obs['velocity'] = physics.velocity()
        limb_offsets = []
        for sensor_name in physics.gen.sensors:
            geom_name = sensor_name.strip('_sensor')
            limb_offsets.append(physics.geom_offset("root", geom_name))
        obs['limb_offsets'] = np.hstack(limb_offsets)
        return obs
    
    def get_reward(self, physics):
        radii = physics.named.model.geom_size[['root', 'target'], 0].sum()
        in_target = rewards.tolerance(np.linalg.norm(physics.geom_offset('root', 'target')),
                                      bounds=(0, radii), margin=2*radii)
        is_upright = physics.upright()
        return (7 * in_target + is_upright) / 8
        
    
    

