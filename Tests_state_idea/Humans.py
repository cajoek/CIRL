"""
The different behaviours of the human. Takes observations, states and returns an action.
"""
import numpy as np


# H plays any optimal action with uniform probability.
class Optimal:
    def __init__(self, Q):
        self.Q = Q
        self._name = 'H_Optimal'

    def act(self, state):
        action_score = np.max(self.Q[state, :, :], axis=1)
        aH = np.random.choice(np.flatnonzero(action_score == action_score.max()))
        return aH

    def observe(self, state, aR):
        pass

    def reset(self):
        pass

    def get_name(self):
        return self._name


# The human acts stochastically in accordance with the Q-value of the action
class ApproxOpt:
    def __init__(self, Q, eta=1.0):
        self.Q = Q
        self.eta = eta
        self._name = 'H_ApproxOpt'

    def act(self, state):
        action_score = np.max(self.Q[state, :, :], axis=1)
        action_prob = softmax(action_score, self.eta)
        aH = np.random.choice(action_prob.size, p=action_prob)
        return aH

    def observe(self, state, aR):
        pass

    def reset(self):
        pass

    def get_name(self):
        return self._name + r':$\eta$=' + str(self.eta)


# The human avoids optimal actions that can teach the AI more later.
class Teaching:
    def __init__(self, Q):
        self.Q = Q
        self._name = 'H_Teaching'

    def act(self, state):
        action_score = np.max(self.Q[state, :, :], axis=1)
        optimal_actions = (action_score == action_score.max())
        if np.count_nonzero(optimal_actions) > 1 and optimal_actions[1] != 0:
            pos = np.array(np.unravel_index(state, (np.array([5, 5, 5, 1])+1)))
            if pos[3] == 0:
                optimal_actions[2] = 0
            else:
                optimal_actions[0] = 0
        aH = np.random.choice(np.flatnonzero(optimal_actions))
        return aH

    def observe(self, state, aR):
        pass

    def reset(self):
        pass

    def get_name(self):
        return self._name


# The human acts stochastically in accordance with the Q-value of the action
class IndeOpt:
    def __init__(self, Q, eta=1.0):
        self.Q = Q
        self.n_aH = Q.shape[1]
        self.n_aR = Q.shape[2]
        self.eta = eta
        self._name = 'H_IndeOpt'

    def act(self, state):
        temp = np.max(self.Q[state, :, :], axis=0)
        optimal_aR = np.flatnonzero(temp == temp.max())
        action_score = np.zeros(self.n_aH)
        for aH in range(self.n_aH):
            for aR in optimal_aR:
                action_score[aH] += 1/optimal_aR.size*self.Q[state, aH, aR]

        action_prob = softmax(action_score, self.eta)
        aH = np.random.choice(action_prob.size, p=action_prob)
        return aH

    def observe(self, state, aR):
        pass

    def reset(self):
        pass

    def get_name(self):
        return self._name + r':$\eta$=' + str(self.eta)


class Repeating:
    def __init__(self, Q, eta=1.0):
        self.Q = Q
        self.eta = eta
        self.last_state_action = np.zeros(Q.shape[0], dtype=int)
        self._name = 'H_Repeating'

    def act(self, state):
        last_aR = self.last_state_action[state]
        action_score = self.Q[state, :, last_aR]
        action_prob = softmax(action_score, self.eta)
        aH = np.random.choice(action_prob.size, p=action_prob)
        return aH

    def observe(self, state, aR):
        self.last_state_action[state] = aR

    def reset(self):
        self.last_state_action[:] = 0

    def get_name(self):
        return self._name + r':$\eta$=' + str(self.eta)


class FreqRep:
    def __init__(self, Q, eta=1.0):
        self.Q = Q
        self.n_aH = Q.shape[1]
        self.n_aR = Q.shape[2]
        self.eta = eta
        self.freq_state_action = np.zeros((Q.shape[0], Q.shape[2]), dtype=int)
        self._name = 'H_FreqRep'

    def act(self, state):
        action_freq = self.freq_state_action[state, :]
        if np.all(action_freq == 0):
            action_freq[:] = 1

        action_score = np.zeros(self.n_aH)
        for aH in range(self.n_aH):
            for aR in range(self.n_aR):
                action_score[aH] += self.Q[state, aH, aR] * action_freq[aR]/np.sum(action_freq)

        action_prob = softmax(action_score, self.eta)
        aH = np.random.choice(action_prob.size, p=action_prob)
        return aH

    def observe(self, state, aR):
        self.freq_state_action[state, aR] += 1

    def reset(self):
        self.freq_state_action[:] = 0

    def get_name(self):
        return self._name + r':$\eta$=' + str(self.eta)


def softmax(array, eta=1.0):
    value = np.exp(eta * array)
    return value/np.sum(value)
