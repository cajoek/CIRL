"""
The different learners. Takes observations, states and returns an action.
"""
import numpy as np


# AI only has an assumption about if the human thinks the AI is
# repeating or not and how optimal it thinks the human is.
class ApproximatelyOptimal:
    def __init__(self, Q, ai_type='optimal', eta=1.0, delta=0.02):
        self.Q = Q
        assert ai_type == 'optimal' or ai_type == 'repeating' or ai_type == 'freq_rep'
        self._ai_type = ai_type
        self.eta = eta
        self.delta = delta

        self.n_theta = Q.shape[-1]
        self.eta = eta
        self.phi = np.ones(self.n_theta) / self.n_theta
        self.phi_prev = np.zeros(self.n_theta)

        self.n_exploration_steps = 0
        self.n_explored_steps = 0
        self.explore_theta = None

        self.n_aH = Q.shape[1]
        self.n_aR = Q.shape[2]
        self.last_state_action = None  # Keeps track of the last action R made in that state
        self.freq_state_action = None  # Keeps track of the number of times that R made an action
        if self._ai_type == 'repeating':
            self.last_state_action = np.zeros(Q.shape[0], dtype=int)
        elif self._ai_type == 'freq_rep':
            self.freq_state_action = np.zeros((Q.shape[0], Q.shape[2]), dtype=int)

    def act(self, state):
        argmax = np.argmax(self.phi)
        if self.n_explored_steps < self.n_exploration_steps or (np.max(self.phi) <= 0.5 and self.phi[argmax] <= 0.02 + self.phi_prev[argmax]):  # Explore

            if self.n_explored_steps >= self.n_exploration_steps:
                # Chose a new theta to explore
                self.explore_theta = np.random.choice(self.n_theta, p=self.phi)
                self.n_explored_steps = 0
                if self.n_exploration_steps == 0:
                    self.n_exploration_steps = 1
                else:
                    self.n_exploration_steps *= 2

            action_score = np.max(self.Q[state, :, :, self.explore_theta], axis=0)
            aR = np.random.choice(np.flatnonzero(action_score == action_score.max()))
            self.n_explored_steps += 1

        else:  # Exploit
            action_score = np.zeros(self.n_aR)
            P = self.get_likelihood(state)
            for aR in range(self.n_aR):
                for theta in range(self.n_theta):
                    for aH in range(self.n_aH):
                        action_score[aR] += self.Q[state, aH, aR, theta]*P[aH, theta]*self.phi[theta]
            aR = np.random.choice(np.flatnonzero(action_score == action_score.max()))
            self.explore_theta = None
            self.n_exploration_steps = 0
            self.n_explored_steps = 0

        return aR

    def observe(self, state, aH, aR):
        self.phi_prev = self.phi.copy()
        self.phi *= self.get_likelihood(state)[aH]
        self.phi /= np.sum(self.phi)

        if self._ai_type == 'repeating':
            self.last_state_action[state] = aR
        elif self._ai_type == 'freq_rep':
            self.freq_state_action[state, aR] += 1

    def get_likelihood(self, state):
        P = np.zeros((self.n_aH, self.n_theta))
        if self._ai_type == 'optimal':
            for aH in range(self.n_aH):
                for theta in range(self.n_theta):
                    P[aH, theta] = np.max(softmax(self.Q[state, aH, :, theta], self.eta))

        elif self._ai_type == 'repeating':
            aR_assumed = self.last_state_action[state]
            for aH in range(self.n_aH):
                for theta in range(self.n_theta):
                    P[aH, theta] = softmax(self.Q[state, aH, :, theta], self.eta)[aR_assumed]

        elif self._ai_type == 'freq_rep':
            action_freq = self.freq_state_action[state, :]
            if np.all(action_freq == 0):
                action_freq[:] = 1
            f = self.freq_state_action[state, :] / np.sum(self.freq_state_action[state, :], dtype=float)

            for aH in range(self.n_aH):
                for theta in range(self.n_theta):
                    #f = self.freq_state_action[state, :]/np.sum(self.freq_state_action[state, :], dtype=float)
                    P[aH, theta] = np.max(softmax(self.Q[state, aH, :, theta]*f, self.eta))
        return P


    def get_theta_prob(self, theta):
        return self.phi[theta]

    def reset(self):
        self.phi = np.ones(self.n_theta) / self.n_theta
        self.phi_prev = np.zeros(self.n_theta)
        self.explore_theta = None
        self.n_exploration_steps = 0
        self.n_explored_steps = 0
        if self._ai_type == 'repeating':
            self.last_state_action[:] = 0
        elif self._ai_type == 'freq_rep':
            self.freq_state_action[:] = 0

    def get_name(self):
        return r'R:$\eta$=' + str(self.eta)


def softmax(array, eta=1.0):
    value = np.exp(eta * array)
    return value/np.sum(value)
