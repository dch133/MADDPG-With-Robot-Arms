import gym
from gym import error


try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here:"
        " https://github.com/openai/mujoco-py/.)".format(e))

'''
Set this up in the reward section below

def _is_collision(self, agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False

def _reward(self, agent, world):
    # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    rew = 0
    for l in world.landmarks:
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        rew -= min(dists)
    if agent.collide:
        for a in world.agents:
            if self.is_collision(a, agent):
                rew -= 0.5
    return rew
'''


class TwoArmsEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, num_env):
        self.envs = []
        for _ in range(num_env):
            self.envs.append(gym.make(env_id))

    def reset(self):
        for env in self.envs:
            env.reset()

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        for env, ac in zip(self.envs, actions):
            ob, rew, done, info = env.step(ac)
            obs.append(ob)
            dones.append(done)
            infos.append(info)

            # Calculate reward and penalize collisions
            # TODO
            rewards.append(rew)

            if done:
                env.reset()

        return obs, rewards, dones, infos
