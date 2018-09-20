"""
Gives the environment a learner, a human type and monitors the learning.
"""
from Tests_simple_idea.Environment import SuperSimpleGame
import Tests_simple_idea.Humans as humans
import Tests_simple_idea.Robots as robots
import numpy as np
import matplotlib.pyplot as plt


def simulate(n_simulations=10000, n_time_steps=60):
    env = SuperSimpleGame(reduced_theta=4)
    Q = env.get_Q()
    agents = [(robots.ApproximatelyOptimal(Q, 'optimal', 1.0), humans.IndeOpt(Q, 1.0)),
              (robots.ApproximatelyOptimal(Q, 'optimal', 1.0), humans.Optimal(Q)),
              (robots.ApproximatelyOptimal(Q, 'optimal', 1.0), humans.Teaching(Q))]


    n_tests = len(agents)
    theta_prob = np.zeros((n_tests, n_simulations, n_time_steps+1))
    completion_time = np.zeros((len(agents), n_simulations))
    completion_time[:, :] = n_time_steps
    set_ = 0
    for robot, human in agents:
        for i in range(n_simulations):
            state = env.reset()
            theta = env.theta
            robot.reset()
            human.reset()
            theta_prob[set_, i, 0] = robot.get_theta_prob(theta)
            #env.render()

            for t in range(1, n_time_steps+1):
                aR = robot.act(state)
                aH = human.act(state, theta)
                robot.observe(state, aH, aR)
                human.observe(state, aR)
                #print(robot.phi)

                state, is_done = env.step(aH, aR)
                if is_done and t < completion_time[set_, i]:
                    theta_prob[set_, i, t:] = robot.get_theta_prob(theta)
                    completion_time[set_, i] = t
                    #env.render()
                    #break

                theta_prob[set_, i, t] = robot.get_theta_prob(theta)
                #env.render()
        set_ += 1

    plt.figure(1)
    color = 0
    for i in range(n_tests):
        r, h = agents[i]
        label = r.get_name() + ' ' + h.get_name()
        plot_result(theta_prob[i, :, :], label, color, True)
        color += 1
    plt.gcf().set_size_inches(10, 6)
    plt.legend(loc='lower right')

    median_completion_time = np.median(completion_time, axis=1)
    print(median_completion_time)
    max_prob = np.max(theta_prob, axis=2)
    print(np.mean(max_prob >= 0.5, axis=1))

    plt.xlabel('Time')
    plt.ylabel('Confidence')
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