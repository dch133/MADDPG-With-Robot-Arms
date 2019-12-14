from gym import error
from gym_two_arms.envs.two_arms_env import TwoArmsEnv

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here:"
        " https://github.com/openai/mujoco-py/.)".format(e))


class TwoArmsEnvSymmetric(TwoArmsEnv):
    """
    TwoArmsEnvSymmetric this environment allows all robot arms to perform the exact same movement

    """

    def __init__(self):
        super(TwoArmsEnv, self).__init__()

    def step(self, action):
        obs = []
        rewards = []
        dones = []
        infos = []

        for env in self.envs:
            ob, rew, done, info = env.step(action)
            obs.append(ob)
            dones.append(done)
            infos.append(info)
            rewards.append(rew)

            if done:
                env.reset()

        return obs, rewards, dones, infos
