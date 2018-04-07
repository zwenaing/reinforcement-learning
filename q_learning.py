from abstract_learning import AbstractLearning
import numpy as np

class QLearning(AbstractLearning):

    def __init__(self, size=5, discount_factor=0.99, n_experiments=500, n_episodes=500, epsilon=0.1, alpha=0.1, random_seed=None):
        super(QLearning, self).__init__(size, discount_factor, n_experiments, n_episodes, epsilon, alpha, random_seed)

    def fit(self):
        q_values_sum = []
        for i in range(self.size):
            q_values_sum.append([])
            for j in range(self.size):
                q_values_sum[i].append({"up": 0., "down": 0., "left": 0., "right": 0.})

        steps = []
        for j in range(self.n_experiments):
            print("Experiment: ", j)
            self.reset_q_states()
            steps.append([])
            for i in range(self.n_episodes):
                current_row = self.size - 1
                current_col = 0
                while not (current_col == self.size - 1 and current_row == 0):
                    # pi(s) = max Q-value(s, a)
                    max_value, max_action = self.get_max_q_state(current_row, current_col)
                    # T(s, a, s')
                    max_action = self.get_action(max_action)
                    # s'
                    new_row, new_col = self.get_new_state(current_row, current_col, max_action)
                    # R(s, a, s') reward obtained at the new state
                    reward = self.rewards[new_row][new_col]
                    # Q(s, a)
                    q_value = self.q_values[current_row][current_col]
                    current_q_value = q_value[max_action]
                    # max Q(s', a)
                    next_max_value, next_max_action = self.get_max_q_state(new_row, new_col)
                    # Q'(s, a) = R + gamma * max Q(s', a)
                    new_q_value = reward + (self.discount_factor * next_max_value)
                    # Q(s, a) = Q(s, a) + alpha * (Q'(s,a) - Q(s,a))
                    self.q_values[current_row][current_col][max_action] = current_q_value + self.alpha * (new_q_value - current_q_value)
                    # update the current row
                    current_row, current_col = new_row, new_col
                self.update_policy()
                eps_steps = self.num_steps()
                steps[j].append(eps_steps)

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

        # steps /= self.n_experiments

        return self.q_values, steps

    def f(self, n_exp):
        qlearning = QLearning(n_experiments=n_exp, size=self.size, discount_factor=self.discount_factor,
                              n_episodes=self.n_episodes, epsilon=self.epsilon, alpha=self.alpha)
        return qlearning.fit()


if __name__ == '__main__':
    qlearning = QLearning(alpha=0)
    steps = qlearning.fit_threads(10)
    steps = np.array(steps)
    avg_steps = np.average(steps, axis=0)
    print(avg_steps.shape)
    print(avg_steps)
    np.save("alpha0.npy", avg_steps)
    print(np.load("alpha0.npy"))