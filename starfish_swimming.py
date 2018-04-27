import os
os.environ['DISPLAY'] = ':1'
import numpy as np

import numpy as np
from skimage import io
from dm_control.suite import common

from genome_synth import Genesis, prettify
from findtarget import FindTarget, Physics

from dm_control import mujoco
from dm_control.rl import control
from dm_control.rl.control import PhysicsError

from xml.etree import ElementTree as ET

from tensorforce.agents import PPOAgent
from tensorforce.agents import RandomAgent

import tqdm

class DummyGen(Genesis):
    def override_from_xml(self, xml_str):
        self.joints = set()
        spec = ET.fromstring(xml_str)
        for motor in spec.iter('motor'):
            if 'joint' in motor.attrib:
                self.joints.add(motor.attrib['joint'])

def observation2state(obs):
    result = np.array([])
    for(_, data) in obs.items():
        result = np.append(result, data)
    return result

def main(xml_name):
    with open(xml_name) as xmlf:
        xml_str = xmlf.read()

    gen = DummyGen()
    gen.override_from_xml(xml_str)

    _DEFAULT_TIME_LIMIT = 10
    _CONTROL_TIMESTEP = .04
    display_stride = 1 / .04 // 24

    genesis_physics = Physics.from_xml_string(common.read_model(os.path.join(os.getcwd(), xml_name)), 
                                              common.ASSETS)

    genesis_physics.set_genesis(gen)
    genesis_task = FindTarget()
    genesis_env = control.Environment(genesis_physics, 
                                     genesis_task,
                                     control_timestep=_CONTROL_TIMESTEP,
                                     time_limit=_DEFAULT_TIME_LIMIT)
    action_spec = genesis_env.action_spec()
    observation_spec = genesis_env.observation_spec()
    observation_shape = np.array([0])

    for (name, row) in observation_spec.items():
        print (name, observation_shape, row.shape)
        if(row.shape == ()):
            observation_shape[0] += 1
            continue
        print(row.shape)
        observation_shape[0] += row.shape[0]
    observation_shape = (observation_shape[0],)
    print(action_spec)
    print(action_spec.minimum)
    agent = PPOAgent(
        states=dict(type='float', min_value=action_spec.minimum, max_value=action_spec.maximum, shape=observation_shape),
        actions=dict(type='float', min_value=action_spec.minimum, max_value=action_spec.maximum, shape=action_spec.shape),
        network=[
            dict(type='dense', size=128, activation='relu'),
            dict(type='dense', size=64, activation='relu'),
            dict(type='dense', size=16, activation='tanh')
        ],
        step_optimizer={
            "type": "adam",
            "learning_rate": 1e-4
        },
        entropy_regularization=0.01,
        batching_capacity=64,
        subsampling_fraction=0.1,
        optimization_steps=50,
        discount=0.99,
        likelihood_ratio_clipping=0.2,
        baseline_mode="states",
        baseline={
            "type":"mlp",
            "sizes": [32, 32]
        },
        baseline_optimizer={
            "type":"multi_step",
            "optimizer": {
                "type": "adam",
                "learning_rate": 1e-4
            },
            "num_steps": 5
        },
        update_mode={
            "unit": "episodes",
            "batch_size": 128,
            "frequency": 10
        },
        memory={
            "type": "latest",
            "include_next_states": False,
            "capacity": 2000
        }
    )

    time_step = genesis_env.reset()
    curtime = 0.0
    top_view = genesis_env.physics.render(480, 480, camera_id='tracking_top')
    side_view = genesis_env.physics.render(480, 480, camera_id='arm_eye')
    did_except = False
    
    NUM_EPISODES = 10000
    N_INPROG_VIDS = 4
    VID_EVERY = NUM_EPISODES // N_INPROG_VIDS

    for i in tqdm.tqdm(range(NUM_EPISODES)):
        time_step = genesis_env.reset()
        j = 0
        tot = 0
        reward = []
        while not time_step.last():
            state = observation2state(time_step.observation)
            action = agent.act(state)
            time_step = genesis_env.step(action)
            tot += time_step.reward
            reward.append(time_step.reward)
            agent.observe(reward=time_step.reward, terminal=time_step.last())
            if(j % 50 == 0 and i % 25 == 1):
                pass
                #clear_output()
                #img = plt.imshow(np.array(env.physics.render(480, 640)).reshape(480, 640, 3))
                #plt.pause(0.5)
                
            j += 1

        if i % 100 == 0:
                #tot /= j
            tqdm.tqdm.write("for episode " + str(i) +  " : " + str(tot))
            

        if (i % VID_EVERY) == 0 or i == NUM_EPISODES - 1:
            
            agent.save_model('./models/starfish_model_target')

            time_step = genesis_env.reset()
            
            vid_suffix = str(i)
            if i == NUM_EPISODES - 1:
                vid_suffix = 'final'
            vid_name = 'videos/starfish_{}.mp4'.format(vid_suffix)

            imnames = set()
            picidx = 0
            curtime = 0.0

            while not time_step.last():
                try:
                    state = observation2state(time_step.observation)
                    action = agent.act(state)
                    time_step = genesis_env.step(action)
                    savename = "/tmp/starfish_{0:04}.jpg".format(picidx)
                    picidx += 1
                    imnames.add(savename)
                    curtime += _CONTROL_TIMESTEP
                    top_view = genesis_env.physics.render(480, 480, camera_id='tracking_top')
                    side_view = genesis_env.physics.render(480, 480, camera_id='arm_eye')
                    #plt.imshow(np.concatenate((top_view, side_view), axis=1))
                    #plt.pause(0.5)
                    io.imsave(savename, np.concatenate((top_view, side_view), axis=1))
                except PhysicsError:
                    print('except')
                    did_except = True
                    break
            if os.path.isfile(vid_name):
                os.remove(vid_name)
            if not did_except:
                os.system('ffmpeg -nostats -loglevel 0 -f image2 -pattern_type sequence -i "/tmp/starfish_%4d.jpg" -qscale:v 0 {}'.format(vid_name))
            for name in imnames:
                os.remove(name)
            print("recorded video")

if __name__ == '__main__':
    main('./starfish_world.xml')