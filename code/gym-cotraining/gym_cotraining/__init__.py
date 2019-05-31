from gym.envs.registration import register

register(
    id='cotraining-v0',
    entry_point='gym_cotraining.envs:CoTraining',
)