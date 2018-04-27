from gym.envs.registration import register

register(
    id='Starfish-SwimX-v0',
    entry_point='starfish.starfish:SwimX'
)

register(
    id='Starfish-SwimXLarge-v0',
    entry_point='starfish.starfish:SwimX',
    kwargs={'large': True}
)

register(
    id='Starfish-FindTarget-v0',
    entry_point='starfish.starfish:FindTarget'
)

register(
    id='Starfish-FindTargetLarge-v0',
    entry_point='starfish.starfish:FindTarget',
    kwargs={'large': True}
)

register(
    id='Starfish-FindTargetHER-v0',
    entry_point='starfish.starfish:FindTargetHER'
)