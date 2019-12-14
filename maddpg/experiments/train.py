import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, 1):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = 0
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')

                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


def train_one_arm(arglist, static_goal):
    with U.single_threaded_session():

        # Create environment
        two_arms = create_2_arm_env()
        left_env = two_arms[0]
        left_env.reset()
        # Create agent trainers
        obs_shape_n = [left_env.observation_space.shape]  # env.n = 1
        num_adversaries = 0
        left_env.action_space = [left_env.action_space]  # To make the types match for maddpg.py
        trainers = get_trainers(left_env, num_adversaries, obs_shape_n, arglist)
        left_env.action_space = left_env.action_space[0]

        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0]]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n_left = left_env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        goal = left_env.unwrapped.goal

        f = open("/home/daniel-ubuntu/Documents/myfile.txt", "w")

        print('Starting iterations...')
        f.write('Starting iterations...\n')
        while True:
            # get action
            obs = obs_n_left
            agent = trainers[0]
            action_n_left = agent.action(obs)

            # environment step
            episode_step += 1
            if episode_step >= 44000:
                time.sleep(.004)  # Slow down simulation to avoid following error:
                # mujoco_py.builder.MujocoException: Got MuJoCo Warning: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable.
            new_obs_n_left, rew_n_left, done_n_left, info_n_left = left_env.step(action_n_left)

            done = done_n_left
            terminal = (episode_step >= arglist.max_episode_len)

            # collect experience
            agent_left = trainers[0]
            agent_left.experience(obs_n_left, action_n_left, rew_n_left, new_obs_n_left, done_n_left, terminal)
            obs_n_left = new_obs_n_left

            episode_rewards[-1] += rew_n_left
            agent_rewards[0][-1] += rew_n_left

            if done or terminal:
                obs_n_left = left_env.reset()
                if static_goal:
                    left_env.unwrapped.goal = goal

                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                agent_info[-1][0].append(info_n_left['i_success'])

                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    f.write('Finished benchmarking, now saving...\n')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                left_env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                    f.write("steps: {}, episodes: {}, mean episode reward: {}, time: {}\n".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                    f.write(
                        "steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}\n".format(
                            train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                            [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards],
                            round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                f.write('...Finished total of {} episodes.\n'.format(len(episode_rewards)))
                break


def train_two_arms(arglist, static_goal):
    with U.single_threaded_session():

        # Create environment
        two_arms = create_2_arm_env()
        left_env = two_arms[0]
        right_env = two_arms[1]
        left_env.reset()
        # Create agent trainers
        obs_shape_n = [left_env.observation_space.shape]  # env.n = 1
        num_adversaries = 0
        left_env.action_space = [left_env.action_space]  # To make the types match for maddpg.py
        trainers_left = get_trainers(left_env, num_adversaries, obs_shape_n, arglist)
        left_env.action_space = left_env.action_space[0]

        trainers_right = trainers_left

        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0]]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n_left = left_env.reset()
        obs_n_right = right_env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        goal = left_env.unwrapped.goal

        print('Starting iterations...')
        while True:
            # get action
            obs = obs_n_left
            agent = trainers_left[0]
            action_n_left = agent.action(obs)

            obs = obs_n_right
            agent = trainers_right[0]
            action_n_right = agent.action(obs)

            # environment step
            episode_step += 1
            if episode_step >= 44000:
                time.sleep(.004) # Slow down simulation to avoid following error:
                # mujoco_py.builder.MujocoException: Got MuJoCo Warning: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable.
            new_obs_n_left, rew_n_left, done_n_left, info_n_left = left_env.step(action_n_left)
            new_obs_n_right, rew_n_right, done_n_right, info_n_right = right_env.step(action_n_right)
            done = done_n_left or done_n_right
            terminal = (episode_step >= arglist.max_episode_len)

            # collect experience
            agent_left = trainers_left[0]
            agent_left.experience(obs_n_left, action_n_left, rew_n_left, new_obs_n_left, done_n_left, terminal)
            obs_n_left = new_obs_n_left

            agent_right = trainers_right[0]
            agent_right.experience(obs_n_right, action_n_right, rew_n_right, new_obs_n_right, done_n_right, terminal)
            obs_n_right = new_obs_n_right

            if rew_n_left > rew_n_right:
                episode_rewards[-1] += rew_n_left
                agent_rewards[0][-1] += rew_n_left
            else:
                episode_rewards[-1] += rew_n_right
                agent_rewards[0][-1] += rew_n_right


            if done or terminal:
                obs_n_left = left_env.reset()
                obs_n_right = right_env.reset()
                if static_goal:
                    left_env.unwrapped.goal = goal
                    right_env.unwrapped.goal = goal

                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                agent_info[-1][0].append(info_n_left['i_success'])
                agent_info[-1][1].append(info_n_right['i_success'])

                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                left_env.render()
                right_env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers_left:
                agent.preupdate()
            for agent in trainers_left:
                loss = agent.update(trainers_left, train_step)

            for agent in trainers_right:
                agent.preupdate()
            for agent in trainers_right:
                loss = agent.update(trainers_right, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        round(time.time() - t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


def two_arms_sim():
    import gym

    right_hand_shift = 0.1
    env = gym.make('FetchReach-v1')
    env = env.unwrapped
    env2 = gym.make('FetchReach-v1')
    env2 = env2.unwrapped

    env.reset()
    env2.reset()

    # Set the goal of both arms to the same
    env2.goal = np.copy(env.goal)
    # Set position of env2 slightly on the right of env (mimicking right arm reference frame)
    env2.goal[1] -= right_hand_shift

    for _ in range(1000):
        action = env.action_space.sample()
        env.render()
        env2.render()

        observation, reward, done, info = env.step(action)  # take a random action
        if done:
            print("Left Arm found goal!")
            break
        observation, reward, done, info = env2.step(action)  # take a random action

        if done:
            print("Right Arm found goal!")
            break
    env.close()


# Create 2 Arm env
def create_2_arm_env():
    import gym

    right_hand_shift = 0.1
    env = gym.make('FetchReach-v1')
    env = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])
    env2 = gym.make('FetchReach-v1')
    env2 = gym.wrappers.FlattenDictWrapper(env2, dict_keys=['observation', 'desired_goal'])

    env.reset()
    env2.reset()

    # Remove 'sparse' reward type to get better reward representation
    env.unwrapped.reward_type = ''
    env2.unwrapped.reward_type = ''

    # Set the goal of both arms to the same position
    env2.unwrapped.goal = np.copy(env.unwrapped.goal)
    # Set position of env2 slightly on the right of env (mimicking right arm reference frame)
    env2.unwrapped.goal[1] -= right_hand_shift

    return env, env2

def is_collision(self, agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False

def reward(self, agent, world):
    # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
    rew = 0
    for l in world.landmarks:
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
        rew -= min(dists)
    if agent.collide:
        for a in world.agents:
            if self.is_collision(a, agent):
                rew -= 1
    return rew

def plot_data(in_file, out_file):
    import re

    # text_file = open("/home/daniel-ubuntu/Documents/Data/2arm_300kEps.txt", "r")
    text_file = open(in_file, "r")
    lines = text_file.readlines()

    x = []
    y = []
    for line in lines:
        data = re.split(', |: ', line)
        x.append(data[3])
        y.append(data[5])

    print(x)
    print(y)

    from matplotlib import pyplot as plt

    import numpy as np

    fig, ax = plt.subplots()
    ax.plot(np.array(x).astype(np.float), np.array(y).astype(np.float))
    ax.set_facecolor('xkcd:light grey')
    ax.set(xlabel='episodes', ylabel='mean episode reward')
    ax.grid()

    # fig.savefig("2arm_300kEps.png")
    fig.savefig(out_file)
    # plt.show()

if __name__ == '__main__':
    arglist = parse_args()
    # train(arglist)
    # two_arms_sim()
    train_one_arm(arglist, True)
    # train_two_arms(arglist, True)

    # plot_data("/home/daniel-ubuntu/Documents/Data/7_maddpg_staticgoal_120kEpisodes", "1arm_120kEps.png")
    # plot_data("/home/daniel-ubuntu/Documents/Data/7_maddpg_staticgoal_120kEpisodes", "1arm_120kEps.png")
    # plot_data("/home/daniel-ubuntu/Documents/Data/7_maddpg_staticgoal_120kEpisodes", "1arm_120kEps.png")

