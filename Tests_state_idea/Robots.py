"""
The different learners. Takes observations, states and returns an action.
"""
import numpy as np


# AI only has an assumption about if the human thinks the AI is
# repeating or not and how optimal it thinks the human is.
class ApproximatelyOptimal:
    def __init__(self, q_star, Q_updater, ai_type='optimal', epsilon=0.1):
        self.q_star = q_star
        assert ai_type == 'optimal' or ai_type == 'repeating' or ai_type == 'freq_rep'
        self._ai_type = ai_type
        self.epsilon = epsilon

        self.s_space_size = self.q_star.shape[0]
        self.a_space_H_size = self.q_star.shape[1]
        self.a_space_R_size = self.q_star.shape[2]
        self.theta = np.zeros(self.s_space_size)
        self.nu = np.zeros(self.s_space_size, dtype=int)
        self.Q_updater = Q_updater
        self.Q = np.random.rand(self.s_space_size, self.a_space_H_size, self.a_space_R_size)
        self.n_time_steps = 0

        self.last_state_action = None  # Keeps track of the last action R made in that state
        self.freq_state_action = None  # Keeps track of the number of times that R made an action
        if self._ai_type == 'repeating':
            self.last_state_action = np.zeros(self.s_space_size, dtype=int)
        elif self._ai_type == 'freq_rep':
            self.freq_state_action = np.zeros((self.s_space_size, self.a_space_R_size), dtype=int)

    def act(self, state):
        aR = None
        if np.random.rand() < self.epsilon:
            aR = np.random.choice(self.a_space_R_size)
        else:
            self.update_Q()
            if self._ai_type == 'optimal':
                action_score = np.max(self.Q[state, :, :], axis=0)
                aR = np.random.choice(np.flatnonzero(action_score == action_score.max()))

            elif self._ai_type == 'repeating':
                aR_assumed = self.last_state_action[state]
                action_score = self.Q[state, :, aR_assumed]
                aR = np.random.choice(np.flatnonzero(action_score == action_score.max()))

            elif self._ai_type == 'freq_rep':
                action_freq = self.freq_state_action[state, :]
                if np.all(action_freq == 0):
                    action_freq[:] = 1
                action_score = np.max(self.Q[state, :, :], axis=0)*action_freq
                aR = np.random.choice(np.flatnonzero(action_score == action_score.max()))
        return aR

    def observe(self, state, aH, aR):
        if self._ai_type == 'optimal':
            for s in range(self.s_space_size):
                v = np.max(self.q_star[state, aH, :, s])
                a = np.min(self.q_star[state, :, :, s])
                b = np.max(self.q_star[state, :, :, s])
                if a != b and s != state:
                    alpha = self.nu[s] / (self.nu[s] + 1.0)
                    self.nu[s] += 1
                    self.theta[s] = alpha*self.theta[s] + (1-alpha) * ((v - a) * 2 / (b - a) - 1)  # Linear mapping

        elif self._ai_type == 'repeating':
            for s in range(self.s_space_size):
                v = self.q_star[state, aH, aR, s]
                a = np.min(self.q_star[state, :, aR, s])
                b = np.max(self.q_star[state, :, aR, s])
                if a != b and s != state:
                    alpha = self.nu[s] / (self.nu[s] + 1.0)
                    self.nu[s] += 1
                    self.theta[s] = alpha*self.theta[s] + (1-alpha) * (
                        (v - a) * 2 / (b - a) - 1)  # Linear mapping
            self.last_state_action[state] = aR

        elif self._ai_type == 'freq_rep':
            action_freq = self.freq_state_action[state, :]
            for s in range(self.s_space_size):
                v = np.max(self.q_star[state, aH, :, s]*action_freq)
                a = np.min(self.q_star[state, :, :, s]*action_freq)
                b = np.max(self.q_star[state, :, :, s]*action_freq)
                if a != b and s != state:
                    alpha = self.nu[s] / (self.nu[s] + 1.0)
                    self.nu[s] += 1
                    self.theta[s] = alpha*self.theta[s] + (1-alpha) * (
                        (v - a) * 2 / (b - a) - 1)  # Linear mapping
            self.freq_state_action[state, aR] += 1

        self.n_time_steps += 1

    def update_Q(self):
        self.Q = self.Q_updater(self.Q, self.theta)

    def reset(self):
        self.theta = np.zeros(self.s_space_size)
        self.nu = np.zeros(self.s_space_size, dtype=int)
        self.Q = np.random.rand(self.s_space_size, self.a_space_H_size, self.a_space_R_size)
        if self._ai_type == 'repeating':
            self.last_state_action = np.zeros(self.s_space_size, dtype=int)
        elif self._ai_type == 'freq_rep':
            self.freq_state_action = np.zeros((self.s_space_size, self.a_space_R_size), dtype=int)

    def get_name(self):
        return r'R'


def softmax(array, eta=1.0):
    value = np.exp(eta * array)
    return value/np.sum(value)
