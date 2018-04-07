from abstract_learning import AbstractLearning
from multiprocessing import Pool


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
        steps = 0
        for j in range(self.n_experiments):
            self.reset_q_states()
            steps += self.num_steps()
            for i in range(self.n_episodes):
                self.reset_e_traces()
                # initialize s and a
                current_row = self.size - 1
                current_col = 0
                current_action = "right"

                while not (current_col == self.size - 1 and current_row == 0):
                    # Observe s' and r
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

            for p in range(self.size):
                for q in range(self.size):
                    q_values_sum[p][q]["up"] += self.q_values[p][q]["up"]
                    q_values_sum[p][q]["down"] += self.q_values[p][q]["down"]
                    q_values_sum[p][q]["left"] += self.q_values[p][q]["left"]
                    q_values_sum[p][q]["right"] += self.q_values[p][q]["right"]

        for p in range(self.size):
            for q in range(self.size):
                self.q_values[p][q]["up"] = q_values_sum[p][q]["up"] / self.n_experiments
                self.q_values[p][q]["down"] = q_values_sum[p][q]["down"] / self.n_experiments
                self.q_values[p][q]["left"] = q_values_sum[p][q]["left"] / self.n_experiments
                self.q_values[p][q]["right"] = q_values_sum[p][q]["right"] / self.n_experiments

        steps /= self.n_experiments

        return self.q_values, steps

    def f(self, n_exp):
        slearning = SarsaLearning(n_experiments=n_exp, sarsa_lambda=self.sarsa_lambda, size=self.size,
                                  discount_factor=self.discount_factor, n_episodes=self.n_episodes,
                                  epsilon=self.epsilon, alpha=self.alpha)
        return slearning.fit()


if __name__ == '__main__':
    slearning = SarsaLearning(n_experiments=500, sarsa_lambda=0.5)
    slearning.fit_threads(10)

    for i in range(slearning.size):
        for j in range(slearning.size):
            print("State ", str(i), " ", str(j), ": ", slearning.policy[i][j])
