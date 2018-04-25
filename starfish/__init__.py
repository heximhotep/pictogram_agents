from gym.envs.registration import register

register(
    id='Starfish-SwimX-v0',
    entry_point='starfish.starfish:SwimX'
)

register(
    id='Starfish-FindTarget-v0',
    entry_point='starfish.starfish:FindTarget'
)