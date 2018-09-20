"""
The environment. Takes a joint action and returns a new state.
"""
import numpy as np


class SuperSimpleGame(object):
    def __init__(self, discount=0.9, reduced_theta=4):
        self.discount = discount

        self._a_space_H = np.array([-1, 0, 1])
        self.a_space_H_size = self._a_space_H.shape[0]
        self._a_space_R = np.array([[-1, 0], [0, 0], [1, 0], [0, -1], [0, 1]])
        self.a_space_R_size = self._a_space_R.shape[0]

        self._s_space_min = np.array([0, 0, 0])
        self._s_space_max = np.array([4, 4, 1])
        self.s_space_size = 5*5*2

        # Assign a reward to the reward states
        self.n_theta = 4
        # R[sH, sR, theta]
        self._R = np.zeros((5, 5, 2, self.n_theta))
        self._R[0, 0, 0, 0] = 1
        self._R[0, 4, 0, 1] = 1
        self._R[0, 0, 1, 2] = 1
        self._R[0, 4, 1, 3] = 1
        self.n_theta = reduced_theta  # Can be set to a lower value to simplify the problem

        # Set initial states
        self._position = np.array([2, 2, 0])
        self.theta = np.random.randint(self.n_theta)
        self.time_step = 0

    def _pos_to_state(self):
        return np.ravel_multi_index(self._position, (self._s_space_max + 1))

    def _state_to_pos(self, state):
        return np.unravel_index(state, (self._s_space_max + 1))

    def reset(self):
        self._position = np.array([2, 2, 0])
        self.theta = np.random.randint(self.n_theta)
        self.time_step = 0
        return self._pos_to_state()

    def set_state(self, s):
        self._position = self._state_to_pos(s)

    def step(self, aH, aR):
        self._position += np.append(self._a_space_H[aH], self._a_space_R[aR, :])
        self._position = np.maximum(self._position, self._s_space_min)
        self._position = np.minimum(self._position, self._s_space_max)

        next_state = self._pos_to_state()
        self.time_step += 1
        is_done = bool(self._get_reward()[self.theta])
        #if is_done:
        #    self._position = np.array([2, 2, 0])

        return next_state, is_done

    def _get_reward(self):
        return self._R[self._position[0], self._position[1], self._position[2], :]

    def get_Q(self):
        try:
            return np.load('./Q_' + str(self.discount) + '_.npy')[:, :, :, :self.n_theta]
        except FileNotFoundError:
            Q = np.zeros((self.s_space_size, self.a_space_H_size, self.a_space_R_size, 4))
            for t in range(50):  # approx 50
                for s in range(self.s_space_size):
                    for aH in range(self.a_space_H_size):
                        for aR in range(self.a_space_R_size):
                            for theta in range(self.n_theta):
                                self.set_state(s)
                                s_next = self.step(aH, aR)
                                reward = self._get_reward()
                                v = np.max(Q[s_next, :, :, theta])
                                Q[s, aH, aR, theta] = (reward[theta] + self.discount * v)
            np.save('./Q_' + str(self.discount) + '_', Q)
            self.reset()
            return Q[:, :, :, :self.n_theta]

    def render(self):
        row_0 = ['-'] * 5
        row_1 = ['-'] * 5 + [' theta=' + str(self.theta)]
        row_2 = ['-'] * 5 + [' time step=' + str(self.time_step)]
        row_0[self._position[0]] = '#'
        if self._position[2] == 0:
            row_1[self._position[1]] = '#'
        else:
            row_2[self._position[1]] = '#'
        print("".join(row_0))
        print("".join(row_1))
        print("".join(row_2))
        print("")
