from abstract_learning import AbstractLearning
import numpy as np


class QLearning(AbstractLearning):

    def __init__(self, size=5, discount_factor=0.99, n_experiments=500, n_episodes=500, epsilon=0.1, alpha=0.1, random_seed=None):
        super(QLearning, self).__init__(size, discount_factor, n_experiments, n_episodes, epsilon, alpha, random_seed)

    """
    Train the algorithm using the given parameters.
    """
    def fit(self):
        q_values_sum = []
        for i in range(self.size):
            q_values_sum.append([])
            for j in range(self.size):
                q_values_sum[i].append({"up": 0., "down": 0., "left": 0., "right": 0.})

        steps = []
        max_start_qs = []
        for j in range(self.n_experiments):
            print("Experiment: ", j)
            self.reset_q_states()
            steps.append([])
            max_start_qs.append([])
            for i in range(self.n_episodes):
                current_row = self.size - 1
                current_col = 0
                while not (current_col == self.size - 1 and current_row == 0):
                    # pi(s_learning_startq) = max Q-value(s_learning_startq, a)
                    max_value, max_action = self.get_max_q_state(current_row, current_col)
                    # T(s_learning_startq, a, s_learning_startq')
                    max_action = self.get_action(max_action)
                    # s_learning_startq'
                    new_row, new_col = self.get_new_state(current_row, current_col, max_action)
                    # R(s_learning_startq, a, s_learning_startq') reward obtained at the new state
                    reward = self.rewards[new_row][new_col]
                    # Q(s_learning_startq, a)
                    q_value = self.q_values[current_row][current_col]
                    current_q_value = q_value[max_action]
                    # max Q(s_learning_startq', a)
                    next_max_value, next_max_action = self.get_max_q_state(new_row, new_col)
                    # Q'(s_learning_startq, a) = R + gamma * max Q(s_learning_startq', a)
                    new_q_value = reward + (self.discount_factor * next_max_value)
                    # Q(s_learning_startq, a) = Q(s_learning_startq, a) + alpha * (Q'(s_learning_startq,a) - Q(s_learning_startq,a))
                    self.q_values[current_row][current_col][max_action] = current_q_value + self.alpha * (new_q_value - current_q_value)
                    # update the current row
                    current_row, current_col = new_row, new_col
                self.update_policy()    # update the policy
                eps_steps = self.num_steps()    # get the number of steps to reach the goal
                steps[j].append(eps_steps)
                max_start_qs[j].append(self.q_values[self.size - 1][0]["up"])

            # keep a running sum of q_learning_startq values for all experiments
            for p in range(self.size):
                for q in range(self.size):
                    q_values_sum[p][q]["up"] += self.q_values[p][q]["up"]
                    q_values_sum[p][q]["down"] += self.q_values[p][q]["down"]
                    q_values_sum[p][q]["left"] += self.q_values[p][q]["left"]
                    q_values_sum[p][q]["right"] += self.q_values[p][q]["right"]

        # average the q_learning_startq values over all experiments
        for p in range(self.size):
            for q in range(self.size):
                self.q_values[p][q]["up"] = q_values_sum[p][q]["up"] / self.n_experiments
                self.q_values[p][q]["down"] = q_values_sum[p][q]["down"] / self.n_experiments
                self.q_values[p][q]["left"] = q_values_sum[p][q]["left"] / self.n_experiments
                self.q_values[p][q]["right"] = q_values_sum[p][q]["right"] / self.n_experiments

        return self.q_values, steps, max_start_qs

    """
    The method to pass to the process pool which trains the algorithm for the given number of experiments.
    """
    def f(self, n_exp):
        qlearning = QLearning(n_experiments=n_exp, size=self.size, discount_factor=self.discount_factor,
                              n_episodes=self.n_episodes, epsilon=self.epsilon, alpha=self.alpha)
        return qlearning.fit()


if __name__ == '__main__':
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
    filenames = ["alpha001.npy", "alpha005.npy", "alpha01.npy", "alpha05.npy", "alpha1.npy"]
    for i in range(len(alphas)):
        qlearning = QLearning(alpha=alphas[i])
        _, max_start_qs = qlearning.fit_threads(10)
        max_start_qs = np.array(max_start_qs)
        max_qs = np.max(max_start_qs, axis=0)
        np.save(filenames[i], max_qs)
        print("Save to ", filenames[i])

    epsilons = [0.1, 0.25, 0.5, 1]
    filenames = ["epsilon01.npy", "epsilon025.npy", "epsilon05.npy", "epsilon1.npy"]
    for i in range(len(epsilons)):
        qlearning = QLearning(epsilon=epsilons[i])
        _, max_start_qs = qlearning.fit_threads(10)
        max_start_qs = np.array(max_start_qs)
        max_qs = np.max(max_start_qs, axis=0)
        np.save(filenames[i], max_qs)
        print("Save to ", filenames[i])

