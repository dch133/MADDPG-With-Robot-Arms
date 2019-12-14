from gym.envs.registration import register

register(
    id='TwoArms-v0',
    entry_point='gym_two_arms.envs:TwoArms',

)


register(
    id='TwoArmsSymmetric-v0',
    entry_point='gym_two_arms.envs:TwoArmsSymmetric',

)
