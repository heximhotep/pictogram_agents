## Usage

1. Put an import of this folder into whereever you're trying to run it
2. Get this folder onto your PYTHONPATH and run it

For instance, this might look like:

```python3
import starfish
import gym

env = gym.make('Starfish-SwimX-v0')
```

And at runtime (for instance, with run_mujoco in the ppo2 baseline):
```bash 
DISPLAY=:1 PYTHONPATH=/home/studio/Documents/aman/pictogram_agents:$PYTHONPATH python run_mujoco.py --env Starfish-SwimX-v0 --num-timesteps 201000 --logdir starfish_logs
```