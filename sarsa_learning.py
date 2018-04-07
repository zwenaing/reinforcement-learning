from abstract_learning import AbstractLearning
from multiprocessing import Pool
import numpy as np


class SarsaLearning(AbstractLearning):

    def __init__(self, size=5, discount_factor=0.99, n_experiments=500, n_episodes=500, epsilon=0.1, alpha=0.1, sarsa_lambda=0., random_seed=None):
        super(SarsaLearning, self).__init__(size, discount_factor, n_experiments, n_episodes, epsilon, alpha, random_seed)
        self.sarsa_lambda = sarsa_lambda

        # initialize all eligibility-traces to zeros
        self.e_traces = []
        for i in range(self.size):
            self.e_traces.append([])
            for j in range(self.size):
                self.e_traces[i].append({"up": 0., "down": 0., "left": 0., "right": 0.})

    def reset_e_traces(self):
        for i in range(self.size):
            for j in range(self.size):
                self.e_traces[i][j]["up"] = 0.
                self.e_traces[i][j]["down"] = 0.
                self.e_traces[i][j]["left"] = 0.
                self.e_traces[i][j]["right"] = 0.

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
            #steps.append([])
            max_start_qs.append([])
            for i in range(self.n_episodes):
                self.reset_e_traces()
                # initialize s_learning_startq and a
                current_row = self.size - 1
                current_col = 0
                current_action = "right"

                while not (current_col == self.size - 1 and current_row == 0):
                    # Observe s_learning_startq' and r
                    new_row, new_col = self.get_new_state(current_row, current_col, current_action)
                    reward = self.rewards[new_row][new_col]

                    # a'
                    max_value, max_action = self.get_max_q_state(new_row, new_col)
                    # epsilon greedy policy for a'
                    max_action = self.get_action(max_action)

                    # Q(S', A')
                    next_q_value = self.q_values[new_row][new_col][max_action]
                    current_q_value = self.q_values[current_row][current_col][current_action]

                    # delta
                    delta = reward + self.discount_factor * next_q_value - current_q_value
                    self.e_traces[current_row][current_col][current_action] += 1

                    for p in range(self.size):
                        for q in range(self.size):
                            for a in self.q_values[p][q].keys():
                                self.q_values[p][q][a] += self.alpha * delta * self.e_traces[p][q][a]
                                self.e_traces[p][q][a] = self.discount_factor * self.sarsa_lambda * self.e_traces[p][q][a]

                    current_row, current_col = new_row, new_col
                    current_action = max_action

                self.update_policy()  # update the policy
                # eps_steps = self.num_steps()  # get the number of steps to reach the goal
                # steps[j].append(eps_steps)
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

    def f(self, n_exp):
        slearning = SarsaLearning(n_experiments=n_exp, sarsa_lambda=self.sarsa_lambda, size=self.size,
                                  discount_factor=self.discount_factor, n_episodes=self.n_episodes,
                                  epsilon=self.epsilon, alpha=self.alpha)
        return slearning.fit()


if __name__ == '__main__':

    alphas = [0.01, 0.05, 0.1, 0.5, 1.0]
    filenames = ["alpha001.npy", "alpha005.npy", "alpha01.npy", "alpha05.npy", "alpha1.npy"]
    for i in range(len(alphas)):
        slearning = SarsaLearning(alpha=alphas[i])
        _, max_start_qs = slearning.fit_threads(10)
        max_start_qs = np.array(max_start_qs)
        max_qs = np.max(max_start_qs, axis=0)
        np.save(filenames[i], max_qs)
        print("Save to ", filenames[i])

    epsilons = [0.1, 0.25, 0.5, 1.0]
    filenames = ["epsilon01.npy", "epsilon025.npy", "epsilon05.npy", "epsilon1.npy"]
    for i in range(len(epsilons)):
        slearning = SarsaLearning(epsilon=epsilons[i])
        _, max_start_qs = slearning.fit_threads(10)
        max_start_qs = np.array(max_start_qs)
        max_qs = np.max(max_start_qs, axis=0)
        np.save(filenames[i], max_qs)
        print("Save to ", filenames[i])

    lambdas = [0., 0.25, 0.5, 0.75, 1.0]
    filenames = ["lambda0.npy", "lambda025.npy", "lambda05.npy", "lambda075.npy", "lambda1.npy"]
    for i in range(len(lambdas)):
        slearning = SarsaLearning(sarsa_lambda=lambdas[i])
        _, max_start_qs = slearning.fit_threads(10)
        max_start_qs = np.array(max_start_qs)
        max_qs = np.max(max_start_qs, axis=0)
        np.save(filenames[i], max_qs)
        print("Save to ", filenames[i])
