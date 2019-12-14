#!/usr/bin/env python
import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

# Two Robot Arms environment
if __name__ == '__main__':
    import gym
    import numpy as np

    right_hand_shift = 0.1
    env = gym.make('FetchReach-v1')
    env = env.unwrapped
    env2 = gym.make('FetchReach-v1')
    env2 = env2.unwrapped

    # Set the goal of both arms to the same
    env2.goal = np.copy(env.goal)
    # Set position of env2 slightly on the right of env (mimicking right arm reference frame)
    env2.goal[1] -= right_hand_shift

    env.render()
    env2.render()

    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    policies2 = [InteractivePolicy(env2, i) for i in range(env.n)]

    # execution loop
    obs_n = env.reset()
    obs_n2 = env2.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        act_n2 = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        for i, policy in enumerate(policies2):
            act_n.append(policy.action(obs_n2[i]))

        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        obs_n2, reward_n2, done_n2, _ = env2.step(act_n2)

        # render all agent views
        env.render()
        env2.render()

        # display rewards
        print("Left Arm reward: %0.3f" % reward_n)
        print("Right Arm reward: %0.3f" % reward_n2)

# MultiAgent Environment
# if __name__ == '__main__':
#     # parse arguments
#     parser = argparse.ArgumentParser(description=None)
#     parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
#     args = parser.parse_args()
#
#     # load scenario from script
#     scenario = scenarios.load(args.scenario).Scenario()
#     # create world
#     world = scenario.make_world()
#     # create multiagent environment
#     env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
#     # render call to create viewer window (necessary only for interactive policies)
#     env.render()
#     # create interactive policies for each agent
#     policies = [InteractivePolicy(env,i) for i in range(env.n)]
#     # execution loop
#     obs_n = env.reset()
#     while True:
#         # query for action from each agent's policy
#         act_n = []
#         for i, policy in enumerate(policies):
#             act_n.append(policy.action(obs_n[i]))
#         # step environment
#         obs_n, reward_n, done_n, _ = env.step(act_n)
#         # render all agent views
#         env.render()
#         # display rewards
#         #for agent in env.world.agents:
#         #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
