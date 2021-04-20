import numpy as np
import gym
from math import floor
from random import choices
import itertools
import matplotlib.pyplot as plt
from time import time
from functools import wraps
import math
from random import choice

GAMMA = 1
EPSILON = 0.01
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

def modify_env(env):
    def new_reset(state=None):
        # elapsed_steps = env._elapsed_steps
        env.orig_reset()
        if state is not None:
            env.env.state = state
            # env._elapsed_steps = elapsed_steps
        return np.array(env.env.state)

    env.orig_reset = env.reset
    env.reset = new_reset
    return env


class QAC:
    def __init__(self, alpha, gamma, env, location_centers, speed_centers ,total_steps,
                 policy_evaluation_step):
        self.ALPHA = alpha
        self.GAMMA = gamma
        self.steps = 0
        self.policy_evaluation_step = policy_evaluation_step
        self.env = env
        self.y_axis = []
        self.total_steps = total_steps
        self.action_space = 3
        self.episodes_ran = 1
        self.var_matrix = np.linalg.inv(np.diag(np.array([0.04, 0.0004])))
        self.feature_combinations = \
            np.array([[a, b] for a, b in itertools.product(location_centers, speed_centers)])
        self.weights = np.random.random((1, len(self.feature_combinations)))
        self.policy_weights = np.random.random((self.action_space, len(self.feature_combinations)))


    def learn(self):
        while self.steps < self.total_steps + 1:
            self.env.reset()
            self.learn_episode()
        return self.y_axis

    def learn_episode(self):
        cur_state = self.env.reset()
        done = False
        step_termination = 500
        cur_steps = 0
        discount = self.GAMMA
        alpha = self.ALPHA
        while not done and cur_steps < step_termination:
            if self.steps % self.policy_evaluation_step == 0:
                self.y_axis.append(self.eval_policy())
            action = self.choose_action(cur_state)
            observation, reward, done, info = self.env.step(action)
            delta = reward + (1.0 - done) * (self.GAMMA * self.estimate_value(observation, self.weights)) \
                                           - self.estimate_value(cur_state, self.weights)
            self.weights += 0.02 * delta * self.gradient(cur_state)
            self.policy_weights[action] += 0.02* delta * self.policy_gradient(cur_state, action)
            discount *= 0.95
            self.steps += 1
            print(self.steps)
            cur_steps += 1
            cur_state = observation
        self.episodes_ran += 1
        self.env.reset()

    def gradient(self, state):
        array = []
        for combination in self.feature_combinations:
            array.append(self.apply_feature(state, combination))
        return np.array(array)

    def policy_gradient(self, state, action):
        action_features = []
        expectation = 0
        features = []
        for combination in self.feature_combinations:
            features.append(self.apply_feature(state, combination))
        features = np.array(features)
        other_actions_expectation = np.zeros(shape=features.shape)
        for j in range(self.action_space):
            if j != action:
                probability = self.estimate_policy(state, j)
                other_actions_expectation += probability * features
        sum = features - other_actions_expectation
        return sum

    def action_indicator(self, j, a):
        if j == a:
            return 1
        else:
            return 0

    def apply_feature(self, state, combination):
        xi = np.subtract(np.transpose(np.array([state[0], state[1]])),
                    np.transpose(np.array([combination[0], combination[1]])))
        return np.exp((-1/2) * np.transpose(xi).dot(self.var_matrix).dot(xi))

    def apply_policy_feature(self, state, action, j, combination):
        xi = np.subtract(np.transpose(np.array([state[0], state[1]])),
                    np.transpose(np.array([combination[0], combination[1]])))
        return self.action_indicator(action, j) * np.exp((-1/2) * np.transpose(xi).dot(self.var_matrix).dot(xi))


    def estimate_policy(self, state, action):
        features = []
        divisors = []
        for combination in self.feature_combinations:
            feature = self.apply_feature(state, combination)
            features.append(feature)
            for j in range(self.action_space):
                array = []
                for a in range(self.action_space):
                    array.append(feature)
            divisors.append(array)
        features = np.array(features)
        divisors = np.array(divisors)
        # a=np.diag(np.dot(self.policy_weights, divisors))
        divisor = np.sum(np.exp(np.diag((np.dot(self.policy_weights, divisors)))))
        x = np.exp(np.dot(features, self.policy_weights[action]))
        return x/divisor

    def choose_action(self, state):
        action_probabilities = [self.estimate_policy(state, a) for a in range(self.action_space)]
        return choices(range(self.action_space), action_probabilities)[0]


    def estimate_value(self, state, weights):
        features = []
        for combination in self.feature_combinations:
            features.append(self.apply_feature(state, combination))
        return np.dot(weights, features)


    def eval_policy(self):
        value = 0
        for x in range(5):
            reward_sum = 0
            observation = env.reset()
            for t in range(500):
                observation, reward, done, info = \
                    env.step(self.choose_action(observation))
                reward_sum += reward
                if done:
                    break
            value = value + (1/(x+1)) * (reward_sum - value)
        return value


if __name__ == "__main__":
    # create policy, values
    tick = time()
    # # create env
    env = gym.make('MountainCar-v0')
    env.reset()
    modify_env(env)
    env._max_episode_steps = 500
    # initialize parameters
    # alpha - learning rate
    # steps - total number of steps the Q-Learning algorithm will run
    policy_evaluation_step = 1000
    x_axis = []
    steps = 100000
    for y in range(steps + 1):
        if y % policy_evaluation_step == 0:
            x_axis.append(y)
    y_lists = []
    y_lists_names = []
    observation_space = env.observation_space
    pos_high, vel_high = observation_space.high
    pos_low, vel_low = observation_space.low
    train = QAC(alpha=0.02,
                   gamma=1,
                   env=env,
                   location_centers= np.linspace(pos_low * 0.8, pos_high * 0.8, num=4),
                   speed_centers=np.linspace(vel_low * 0.8, vel_high * 0.8, num=8),
                   total_steps=steps,
                   policy_evaluation_step=policy_evaluation_step)
    y_axis = train.learn()
    y_lists.append(y_axis)
    for y in y_lists:
        plt.plot(x_axis, y)
        plt.xlabel("steps")
        plt.ylabel("policy value")
    print('time elapsed: {}'.format(time() - tick))
    plt.show()
    env.close()
