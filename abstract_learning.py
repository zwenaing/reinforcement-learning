from abc import ABC, abstractmethod
import numpy as np
import random
import math
from multiprocessing import Pool

"""
This class represents the abstract class for both QLearning and SarsaLearning classes and contains 
the common methods in both of these classes.
"""


class AbstractLearning(ABC):

    """
    Initialize the instances of classes which inherits this class.
    """
    def __init__(self, size=5, discount_factor=0.99, n_experiments=500, n_episodes=500, epsilon=0.1, alpha=0.1, random_seed=None):
        # initialize all the parameters of Markov Decision Processes
        self.size = size    # the size of the grid world
        self.discount_factor = discount_factor  # the discount factor
        self.n_experiments = n_experiments  # the number of experiments to run
        self.n_episodes = n_episodes    # the number of episodes per experiment
        self.epsilon = epsilon  # parameter for epsilon greedy policy
        self.alpha = alpha  # the learning rate
        self.random_seed = random_seed      # the random seed for epsilon greedy policy

        # action space
        self.actions = ["up", "down", "left", "right"]

        # all rewards are - 0.1 except the upper right corner where the reward is +5
        # initialize rewards for the state space
        self.rewards = [[- 0.1 for _ in range(self.size)] for _ in range(self.size)]
        self.rewards[0][self.size - 1] = 5

        # initialize all q-values to zeros
        self.q_values = []
        for i in range(self.size):
            self.q_values.append([])
            for j in range(self.size):
                self.q_values[i].append({"up": 0., "down": 0., "left": 0., "right": 0.})

        # randomly initialize policy, essentially state-action pairs
        self.policy = []
        random.seed(a=random_seed)
        for i in range(self.size):
            self.policy.append([])
            for j in range(self.size):
                rand_int = random.randint(0, len(self.actions) - 1)
                self.policy[i].append(self.actions[rand_int])

    """
    Given the current state, return the maximum value and action that maximizes the value.
    :param row the row on the grid
    :param col the column on the grid 
    """
    def get_max_q_state(self, row, col):
        q_values = self.q_values[row][col]
        max_value = float('-Inf')
        max_action = self.actions[0]
        for action in q_values.keys():
            q_value = q_values[action]
            if q_value > max_value:
                max_value = q_value
                max_action = action
        return max_value, max_action

    """
    Reset all the Q values of the grid world to zeros.
    """
    def reset_q_states(self):
        for i in range(self.size):
            for j in range(self.size):
                self.q_values[i][j]["up"] = 0.
                self.q_values[i][j]["down"] = 0.
                self.q_values[i][j]["left"] = 0.
                self.q_values[i][j]["right"] = 0.

    """
    Return the resulting state by taking the given action. 
    """
    def get_new_state(self, current_row, current_col, action):
        new_row = current_row
        new_col = current_col
        if action == "up":
            new_row = current_row - 1
        elif action == "down":
            new_row = current_row + 1
        elif action == "left":
            new_col = current_col - 1
        elif action == "right":
            new_col = current_col + 1

        if new_row < 0 or new_row > self.size - 1:
            new_row = current_row
        if new_col < 0 or new_col > self.size - 1:
            new_col = current_col

        return new_row, new_col

    """
    Randomly determine whether the given action is successfully executed using :param success_rate.
    Returns 1 if succeeded, otherwise 0. 
    """
    def action_success(self, success_rate=0.8):
        return np.random.choice(2, 1, p=[1 - success_rate, success_rate])[0]

    """
    Return the given action if succeeded, otherwise, pick uniformly among four actions.
    :param action the action to execute  
    """
    def get_action(self, action):
        if self.action_success(1. - self.epsilon):
            return action
        else:
            random_action = np.random.choice(4, 1, p=[0.25, 0.25, 0.25, 0.25])[0]
            return self.actions[random_action]

    """
    Initialize the Q values of the grid world to the given Q values.
    :param q_values the given q_values
    """
    def set_q_values(self, q_values):
        for i in range(self.size):
            for j in range(self.size):
                self.q_values[i][j]["up"] = q_values[i][j]["up"]
                self.q_values[i][j]["down"] = q_values[i][j]["down"]
                self.q_values[i][j]["left"] = q_values[i][j]["left"]
                self.q_values[i][j]["right"] = q_values[i][j]["right"]

    """
    Update the current policy using the current Q values of the grid world.
    """
    def update_policy(self):
        for i in range(self.size):
            for j in range(self.size):
                max_value, max_action = self.get_max_q_state(i, j)
                self.policy[i][j] = max_action

    """
    Run the training for multiple experiments on different threads. At the end, average all the
    Q values.
    """
    def fit_threads(self, n_threads):
        # create a process pool
        with Pool(n_threads) as pool:
            thread_q_values = pool.map(self.f, [self.n_experiments // n_threads] * n_threads)
        # initialize the array of dictionaries to keep the running sum of Q values
        q_values_sum = []
        for i in range(self.size):
            q_values_sum.append([])
            for j in range(self.size):
                q_values_sum[i].append({"up": 0., "down": 0., "left": 0., "right": 0.})
        # sum all values
        steps = []
        for p in range(n_threads):
            steps += thread_q_values[p][1]
            #steps.append(thread_q_values[p][1])
            for q in range(self.size):
                for r in range(self.size):
                    q_values_sum[q][r]["up"] += thread_q_values[p][0][q][r]["up"]
                    q_values_sum[q][r]["down"] += thread_q_values[p][0][q][r]["down"]
                    q_values_sum[q][r]["left"] += thread_q_values[p][0][q][r]["left"]
                    q_values_sum[q][r]["right"] += thread_q_values[p][0][q][r]["right"]
        # average all values
        for i in range(self.size):
            for j in range(self.size):
                q_values_sum[i][j]["up"] /= n_threads
                q_values_sum[i][j]["down"] /= n_threads
                q_values_sum[i][j]["left"] /= n_threads
                q_values_sum[i][j]["right"] /= n_threads

        #steps /= n_threads
        # set Q values of the grid world
        self.set_q_values(q_values_sum)
        # update the policy
        self.update_policy()

        #rounded_steps = (int) (math.ceil(steps))
        return steps

    """
    Get the number of steps taken to reach the terminal state from the starting state.
    """
    def num_steps(self):
        steps = 0
        current_row = self.size - 1
        current_col = 0
        while not (current_col == self.size - 1 and current_row == 0):
            # pi(s) = max Q-value(s, a)
            max_value, max_action = self.get_max_q_state(current_row, current_col)
            # T(s, a, s')
            max_action = self.get_action(max_action)
            # s'
            new_row, new_col = self.get_new_state(current_row, current_col, max_action)
            # update the current row
            current_row, current_col = new_row, new_col
            steps += 1

        return steps

