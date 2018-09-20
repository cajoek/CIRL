"""
Gives the environment a learner, a human type and monitors the learning.
"""
from Tests_state_idea.Environment import SuperSimpleGame
import Tests_state_idea.Humans as humans
import Tests_state_idea.Robots as robots
import numpy as np
import matplotlib.pyplot as plt


def simulate(n_simulations=1, n_time_steps=1400):
    env = SuperSimpleGame()
    Q = env.Q
    q_star = env.get_q_star()
    Q_updater = env.update_Q_from_theta

    agents = [(robots.ApproximatelyOptimal(q_star, Q_updater, 'repeating'), humans.Repeating(Q)),
              (robots.ApproximatelyOptimal(q_star, Q_updater, 'freq_rep'), humans.FreqRep(Q))]

    n_tests = len(agents)
    action_error_fraction = np.zeros((n_tests, n_simulations, n_time_steps+1))
    in_right_place = np.zeros((n_tests, n_simulations, 3))

    set_ = 0
    for robot, human in agents:
        for i in range(n_simulations):
            print('Set#:' + str(set_) + ' Simulation#:' + str(i))
            state = env.reset()
            robot.reset()
            human.reset()
            action_error_fraction[set_, i, 0] = env.action_error_fraction(robot.Q)
            # env.render()

            for t in range(1, n_time_steps+1):
                aR = robot.act(state)
                aH = human.act(state)
                robot.observe(state, aH, aR)
                human.observe(state, aR)

                state = env.step(aH, aR)

                action_error_fraction[set_, i, t] = env.action_error_fraction(robot.Q)
                in_right_place[set_, i, 0] += env.in_right_place('H')
                in_right_place[set_, i, 1] += env.in_right_place('R')
                in_right_place[set_, i, 2] += env.in_right_place('B')
                #env.render()
        set_ += 1

    print(np.mean(in_right_place, axis=1)/n_time_steps)

    plt.figure(1)
    color = 0
    for i in range(n_tests):
        r, h = agents[i]
        label = h.get_name()
        plot_result(action_error_fraction[i, :, :], label, color, True)
        color += 1
    plt.gcf().set_size_inches(10, 6)
    plt.legend(loc='upper right')

    plt.xlabel('Time', fontsize=16)
    plt.ylabel('Confidence', fontsize=16)
    plt.show()


def plot_result(theta_probs, label, color=0, plot_average_error=True):
    x = np.arange(theta_probs.shape[1])
    ax = plt.gca()
    mean_theta_prob = np.mean(theta_probs, axis=0)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    ax.plot(mean_theta_prob, label=label, color=colors[color])

    if plot_average_error:
        above = np.zeros_like(mean_theta_prob)
        below = np.zeros_like(mean_theta_prob)
        for i in x:
            slice = theta_probs[:, i]
            above[i] = np.mean(slice[slice >= mean_theta_prob[i]])
            below[i] = np.mean(slice[slice <= mean_theta_prob[i]])
        ax.plot(above, color=colors[color], linestyle=':')
        ax.plot(below, color=colors[color], linestyle=':')


if __name__ == '__main__':
    simulate()
