
# coding: utf-8

# In[111]:


import os
import numpy as np
import quaternion
import xml.etree.cElementTree as ET
import xml.dom.minidom as minidom
import random

from dm_control import mujoco
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.rl import control
from dm_control.rl.control import PhysicsError



import matplotlib.pyplot as plt
from IPython.display import clear_output
import collections
from skimage import io, transform


# In[117]:


class Genesis():
    def __init__(self, max_joints=50, max_sites=8, max_cameras=2):
        self.joints = set()
        self.geoms = set()
        self.sites = set()
        self.cameras = set()
        self.bodies = set()
        self.neighbors = set()
        self.max_joints = max_joints
        self.max_sites = max_sites
        self.max_cameras = max_cameras
        self.rotation_range = [0, 180]
        self.joint_range = [5, 180]
        self.dimension_range = [0.06, 0.16]
        self.name_counter = 0
        
    def norm_val(self, x, _range):
        result = _range[1] - _range[0]
        result *= x
        result += _range[0]
        return result
        
    def norm_rotation(self, x):
        return self.norm_val(x, self.rotation_range)
    
    def norm_jointrange(self, x):
        return self.norm_val(x, self.joint_range)
    
    def norm_dimension(self, x):
        return self.norm_val(x, self.dimension_range)
    
    def mk_joint(self, parent, _name, axis, _range=None):
        if len(self.joints) >= self.max_joints:
            return None
        result = ET.SubElement(parent, 'joint', name=_name, axis="{0} {1} {2}".format(axis[0], axis[1], axis[2]))
        if not _range == None:
            result.set('range', '{0} {1}'.format(_range[0], _range[1]))
        self.joints.add(_name)
        return result
    
    def mk_geom(self, parent, _name, gtype, size, pos=None, quat=None):
        result = ET.SubElement(parent, 'geom', name=_name, size="{0} {1} {2}".format(size[0], size[1], size[2]))
        result.set('type', gtype)
        if pos is not None:
            result.set('pos', '{0} {1} {2}'.format(pos[0], pos[1], pos[2]))
        if quat is not None:
            result.set('quat', '{0} {1} {2} {3}'.format(quat[0], quat[1], quat[2], quat[3]))
        self.geoms.add(_name)
        return result
    
    def mk_site(self, parent, _name, size):
        if len(self.sites) >= self.max_sites:
            return None
        result = ET.SubElement(parent, 'site', name=_name, size=str(size))
        self.sites.add(_name)
        return result
    
    def mk_camera(parent, _name, pos, xaxis, yaxis):
        if len(self.cameras) >= self.max_cameras:
            return None
        result = ET.SubElement(parent, 'camera', name=_name)
        result.set('pos','{0} {1} {2}'.format(pos[0], pos[1], pos[2]))
        result.set('xyaxis','{0} {1} {2} {3} {4} {5}'.format(xaxis[0], xaxis[1], xaxis[2],
                                                             yaxis[0], yaxis[1], yaxis[2]))
        self.cameras.add(_name)
        return result
    
    def mk_body(self, parent, _name, pos=None, quat=None):
        result = ET.SubElement(parent, 'body', name=_name)
        if pos is not None:
            result.set('pos', '{0} {1} {2}'.format(pos[0], pos[1], pos[2]))
        if quat is not None:
            result.set('quat', '{0} {1} {2} {3}'.format(quat[0], quat[1], quat[2], quat[3]))
        self.bodies.add(_name)
        return result
    
    def mk_root(self, parent):
        self.__init__()
        result = ET.SubElement(parent, 'body', name='root')
        result.set('pos', '0 0 0.8')
        result.set('childclass', 'dust')
        light = ET.SubElement(result, 'light', name='light', diffuse='.6 .6 .6')
        light.set('pos', '0 0 0.5')
        light.set('dir', '0 0 -1')
        light.set('specular', '.3 .3 .3')
        light.set('mode', 'track')
        root_joint = ET.SubElement(result, 'joint', name='root', damping='0', limited='false')
        root_joint.set('type','free')
        root_shape = [0.025] * 3
        root_geom = self.mk_geom(result, 'root', 'box', root_shape)
        return result

    def mk_defaults(self, parent):
        result = ET.SubElement(parent, 'default')
        def_motor = ET.SubElement(result, 'motor', ctrlrange="-1 1", ctrllimited="true")
        ET.SubElement(result, 'general', ctrllimited='true')
        def_class = ET.SubElement(result, 'default')
        def_class.set('class','dust')
        def_joint = ET.SubElement(def_class, 'joint')
        def_joint.set('type', 'hinge')
        def_joint.set('range', '-60 60')
        def_joint.set('damping', '.2')
        def_joint.set('stiffness', '.6')
        def_joint.set('armature', '.01')
        def_joint.set('solimplimit', '0 .99 .01')
        
        def_geom = ET.SubElement(def_class, 'geom')
        def_geom.set("friction", ".7")
        def_geom.set("solimp", ".95 .99 .003")
        def_geom.set("solref", ".015 1")
        return result

    def mk_worldbody(self, parent):
        result = ET.SubElement(parent, 'worldbody')
        ET.SubElement(result, 'camera', name='tracking_top', pos='0 0 4', xyaxes='1 0 0 0 1 0', mode='trackcom')
        ground = ET.SubElement(result, 'geom', name='ground', size='.5 .5 .1', material='grid')
        ground.set('type','plane')
        return result
    
    def mk_segment(self, parent, _args):
        name = _args['name']
        rotation = _args['rotation']
        offset = _args['offset']
        dimensions = _args['dimensions']
        joint_range = [-_args['joint_range'], _args['joint_range']]
        
        rotation = quaternion.from_euler_angles(rotation)
        rotation = quaternion.as_float_array(rotation)
        result = self.mk_body(parent, name, pos=offset, quat=rotation)
        
        self.mk_joint(result, name + '_x', [1, 0, 0], _range=joint_range)
        self.mk_joint(result, name + '_y', [0, 1, 0], _range=joint_range)
        self.mk_joint(result, name + '_z', [0, 0, 1], _range=joint_range)
        this_pos = np.array([dimensions[0], 0, 0])
        self.mk_geom(result, name, 'ellipsoid', dimensions, pos=this_pos)
        
        self.neighbors.add((parent.get('name'), _args['name']))
        self.neighbors.add(('root', _args['name']))
        return result, np.array([dimensions[0] * 2, 0, 0])
    
    def mk_actuator_position(self, parent, _name, joint, gear='2'):
        result = ET.SubElement(parent, 'motor', name=_name)
        result.set('joint', joint)
        result.set('gear', gear)
        return result

    def mk_actuators(self, parent):
        result = ET.SubElement(parent, 'actuator')
        for joint in self.joints:
            self.mk_actuator_position(result, joint, joint)
        return result
    
    def mk_morphology(self, parent, _args):
        result, offset = self.mk_segment(parent, _args)
        for childArgs in _args['children']:
            childArgs['offset'] = offset
            child = self.mk_morphology(result, childArgs)
        return result
    
    def mk_leaf(self):
        result = {}
        result['name'] = str(self.name_counter)
        self.name_counter += 1
        result['offset'] = np.zeros(3)
        result['rotation'] = self.norm_rotation(np.random.rand(3))
        longdim = self.norm_dimension(np.random.rand(1)[0])
        shortdims = longdim / (2 + np.random.rand(1)[0] * 4)
        result['dimensions'] = [longdim, shortdims, shortdims]
        result['joint_range'] = self.norm_jointrange(np.random.rand(1))[0]
        result['children'] = []
        result['priority'] = np.random.rand(1)[0]
        return result
    
    def mk_exceptions(self, parent):
        result = ET.SubElement(parent, 'contact')
        for neighs in self.neighbors:
            ET.SubElement(result, 'exclude', body1=neighs[0], body2=neighs[1])
        return result
    
    def combine_trees(self, A, B):
        if str(type(A)) == "<class 'method'>":
            A = A()
        if str(type(B)) == "<class 'method'>":
            B = B()
        aName = A['name']
        bName = B['name']
        if aName == bName:
            bName += 'x'
            B['name'] = bName
        nuname = '{0}_{1}'.format(aName, bName)
        if A['priority'] >= B['priority']:
            A['children'].append(B)
            return A
        else:
            B['children'].append(A)
            return B
    
    '''def mk_morphology(self, parent, _name, offset, curjoints=0):
        aname = '{0}_trunk'.format(_name)
        initjoints = curjoints
        #offset = np.zeros(3)
        result, offset = self.mk_segment(parent, aname, offset)
        exit = False
        idx = 0
        if(len(aname) > 30):
            return result
        while(not exit):
            roll = np.random.rand(1)
            nuname = '{0}_trunk_{1}'.format(_name, idx)
            idx+=1
            if(roll < 0.7 / ((curjoints + 1) / 3)):
                #print("morph branch")
                print("morph branch offset is {0}".format(offset))
                curjoints += 1
                self.mk_morphology(result, nuname, offset, curjoints=curjoints / 2)
                #print(_)
            else:
                exit = True
        return result, offset'''

    def mk_model(self, creation): 
        result = ET.Element('mujoco', model='dust')
        ET.SubElement(result, 'include', file="./common/visual.xml")
        ET.SubElement(result, 'include', file="./common/materials.xml")
        skybox_asset = ET.SubElement(result, 'asset')
        skybox = ET.SubElement(skybox_asset, 'texture', name='skybox', builtin="gradient",
                               rgb1=".4 .6 .8", rgb2="0 0 0", width="800", height="800",
                               mark="random", markrgb="1 1 1")
        skybox.set('type', 'skybox')
        time_option = ET.SubElement(result, 'option', timestep='0.004')
        defaults = self.mk_defaults(result)
        worldbody = self.mk_worldbody(result)
        base_dust = self.mk_root(worldbody)
        #print("about to make morph!")
        #print(creation)
        seggy = self.mk_morphology(base_dust, creation)
        #seggy = self.mk_morphology(base_dust, 'necron', [0,0,0])
        self.mk_exceptions(result)
        self.mk_actuators(result)
        return result


# In[113]:


def prettify(elem):
        """Return a pretty-printed XML string for the Element.
        """
        rough_string = ET.tostring(elem)
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t")


# In[116]:


def get_model_and_assets():
    curpath = os.getcwd()
    return common.read_model(curpath + '/dust.xml'), common.ASSETS

class Physics(mujoco.Physics):
    def offset(self):
        return self.named.data.geom_xpos['root']
    def joint_velocities(self):
        #print(list(self.gen.joints))
        return self.named.data.qvel[list(self.gen.joints)]
    def set_gen(self, gen):
        self.gen = gen

class Avoid(base.Task):
    def __init__(self, random=None):
        super(Avoid, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        #quat = self.random.randn(4)
        #physics.named.data.qpos['root'][3:7] = quat / np.linalg.norm(quat)
        #for joint in physics.gen.joints:
        #    physics.named.data.qpos[joint] = self.random.uniform(-.2, .2)
    
    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['offset'] = physics.offset()
        return obs
    
    def get_reward(self, physics):
        offset_norm = np.linalg.norm(physics.offset())
        energy_penalty = np.linalg.norm(physics.joint_velocities())
        #print("physics offset: {0}\noffset_norm: {1}".format(physics.offset(), offset_norm))
        return offset_norm - energy_penalty ** 2
    


# In[ ]:


'''_DEFAULT_TIME_LIMIT = 4
_CONTROL_TIMESTEP = .04'''


# In[ ]:



'''display_stride = 1 / .04 // 24
counts = 10
picidx = 0
imnames = set()
for i in range(counts):
    thisgen = Genesis()
    tree = ET.fromstring(prettify(thisgen.mk_model()))
    tree = ET.ElementTree(tree)
    tree.write('dust.xml')

    genesis_physics = Physics.from_xml_string(*get_model_and_assets())
    genesis_physics.set_gen(thisgen)
    genesis_task = Avoid()
    genesis_env = control.Environment(genesis_physics, genesis_task, control_timestep=_CONTROL_TIMESTEP, time_limit=_DEFAULT_TIME_LIMIT)

    action_spec = genesis_env.action_spec()
    observation_spec = genesis_env.observation_spec()

    idx = 0
    
    time_step = genesis_env.reset()
    action_bs = np.random.rand(action_spec.shape[0]) * 2 * 3.14
    curtime = 0
    while(not time_step.last()):
        try:
            action = np.sin(action_bs + curtime)
            time_step = genesis_env.step(action)
            curtime += _CONTROL_TIMESTEP
            if idx % display_stride == 0:
                #clear_output()
                #print(type(genesis_env.physics.render(480, 640)))                                                                                                                                                                                                                                                                                                                                                                                   (genesis_env.physics.render(480, 640)).reshape(480, 640, 3)
                savename = "seq_{1:04}.jpg".format(i, picidx)
                imnames.add(savename)
                picidx += 1
                #print(savename)
                io.imsave(savename, genesis_env.physics.render(480, 640))
            idx += 1
        except PhysicsError:
            print('except')
            break'''



# In[ ]:


'''if(os.path.isfile('tomo.mp4')):
    os.remove('tomo.mp4')
!!ffmpeg -f image2 -pattern_type sequence -i "seq_%4d.jpg" -qscale:v 0 tomo.mp4
for name in imnames:
    os.remove(name)'''

