"""
The environment. Takes a joint action and returns a new state.
"""
import numpy as np


class SuperSimpleGame(object):
    def __init__(self, discount=0.9):
        self.discount = discount

        self.a_space_H = np.array([-1, 0, 1])
        self.a_space_H_size = self.a_space_H.shape[0]
        self.a_space_R = np.array([-1, 0, 1])
        self.a_space_R_size = self.a_space_R.shape[0]

        self.a_space_T = np.array([1, -1])

        self.s_space_min = np.array([0, 0, 0, 0])  # Human-Robot-Target-direction
        self.s_space_max = np.array([5, 5, 5, 1])
        self.s_space_size = 6 * 6 * 6 * 2  # 432 states

        self.Q = self.get_Q()

        # Set initial states
        self._position = np.array([3, 3, 0, 0])

    def _pos_to_state(self):
            return np.ravel_multi_index(self._position, (self.s_space_max + 1))

    def _state_to_pos(self, state):
        return np.array(np.unravel_index(state, (self.s_space_max + 1)))

    def reset(self):
        self._position = np.array([3, 3, 0, 0])
        return self._pos_to_state()

    def set_state(self, s):
        self._position = self._state_to_pos(s)

    def step(self, aH, aR):
        if (self._position[2] == self.s_space_min[2] and self._position[3] == 1) or \
                (self._position[2] == self.s_space_max[2] and self._position[3] == 0):
            self._position[3] = np.int64(not self._position[3])
            self._position += np.array([self.a_space_H[aH], self.a_space_R[aR], 0, 0])
        else:
            self._position += np.array([self.a_space_H[aH], self.a_space_R[aR], self.a_space_T[self._position[3]], 0])

        self._position = np.maximum(self._position, self.s_space_min)
        self._position = np.minimum(self._position, self.s_space_max)

        next_state = self._pos_to_state()
        return next_state

    def get_reward(self):
        return float(self._position[0] == self._position[1] == self._position[2])

    def in_right_place(self, agent):
        if agent == 'H':
            return float(self._position[0] == self._position[2])
        elif agent == 'R':
            return float(self._position[1] == self._position[2])
        elif agent == 'B':
            return float(self._position[0] == self._position[1] == self._position[2])

    def get_Q(self):
        try:
            return np.load('./Q_' + str(self.discount) + '_.npy')
        except FileNotFoundError:
            Q = np.zeros((self.s_space_size, self.a_space_H_size, self.a_space_R_size))
            while True:
                Q_prev = Q.copy()
                for s in range(self.s_space_size):
                    for aH in range(self.a_space_H_size):
                        for aR in range(self.a_space_R_size):
                                self.set_state(s)
                                s_next = self.step(aH, aR)
                                reward = self.get_reward()
                                v = np.max(Q[s_next, :, :])
                                Q[s, aH, aR] = (reward + self.discount * v)
                if np.max(np.abs(Q - Q_prev)) < 0.05:
                    break
            np.save('./Q_' + str(self.discount) + '_', Q)
            self.reset()
            return Q

    def get_q_star(self):
        try:
            return np.load('./q_star_' + str(self.discount) + '_.npy')
        except FileNotFoundError:
            q_star = np.zeros((self.s_space_size, self.a_space_H_size, self.a_space_R_size, self.s_space_size))
            for s_goal in range(self.s_space_size):
                Q = np.zeros((self.s_space_size, self.a_space_H_size, self.a_space_R_size))
                for t in range(20):
                    for s in range(self.s_space_size):
                        for aH in range(self.a_space_H_size):
                            for aR in range(self.a_space_R_size):
                                if s == s_goal:
                                    continue
                                self.set_state(s)
                                s_next = self.step(aH, aR)
                                v = np.max(Q[s_next, :, :])
                                # v = np.min(Q[next_state, :, :])
                                reward = 1.0 if s_next == s_goal else 0.0
                                # reward = 0.0 if state == s_goal else 1.0
                                Q[s, aH, aR] = reward + self.discount * v
                q_star[:, :, :, s_goal] = Q
                print(s_goal/float(self.s_space_size))
            np.save('./q_star' + str(self.discount) + '_', q_star)
            self.reset()
            return q_star

    def update_Q_from_theta(self, Q, theta):
        game_state = self._pos_to_state()
        while True:
            Q_prev = Q.copy()
            for s in range(self.s_space_size):
                for aH in range(self.a_space_H_size):
                    for aR in range(self.a_space_R_size):
                        self.set_state(s)
                        s_next = self.step(aH, aR)
                        v = np.max(Q[s_next, :, :])
                        Q[s, aH, aR] = (theta[s_next] + self.discount * v)
            if np.max(np.abs(Q - Q_prev)) < 0.05:
                break
        # Restore state
        self.set_state(game_state)
        return Q

    def action_error_fraction(self, test_Q):
        n_errors = 0
        for s in range(self.s_space_size):
            argmax = np.flatnonzero(self.Q[s, :, :] == np.max(self.Q[s, :, :]))
            n_errors += int(not np.any(argmax == np.argmax(test_Q[s, :, :])))
        return n_errors/self.s_space_size

    def value_difference(self, test_Q):
        pass  # pi = np.argmax(test_Q)

    def render(self):
        row_0 = ['-'] * 6
        row_1 = ['-'] * 6
        row_2 = ['-'] * 6
        row_0[self._position[0]] = '#'
        row_1[self._position[1]] = '#'
        row_2[self._position[2]] = '#'
        dir = ' ->' if self._position[3] < 0.5 else ' <-'
        print("".join(row_0))
        print("".join(row_1) + ' ' + str(self._pos_to_state()))
        print("".join(row_2) + dir)
        print("")
