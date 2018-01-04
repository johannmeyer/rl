""" Main File """
from n_armed_bandit import *
from agent import *

import matplotlib.pyplot as plt
import matplotlib

def plot_reward(reward_list, label):
    """ Helper function for plotting the reward """
    plt.plot(reward_list, label=label)

    ax = plt.gca()
    ax.grid(True)
    ax.set_ylabel('Mean Reward')
    ax.set_xlabel('Steps')
    return ax

def rl_cycle(agent, n_armed_bandit, num_steps):
    """ The cycle between the agent and its environment. """
    reward_list = [0]
    for i in range(num_steps):
        # Choose a one-armed bandit
        action = agent.get_action()

        # Get the reward
        reward = n_armed_bandit.get_reward(action)
        reward_list.append(reward) # for plotting only

        # Update the estimate of the action_value with the reward
        # using the sample-average method i.e. step_size = 1/Ka
        agent.update_q_value(action, reward, 1/agent.Ka[action])
    return reward_list

def repeat_rl_cycle(num_tasks, agent, n_armed_bandit_tasks, num_steps):
    """ Repeat the problem a number of times to get a smooth
    estimate of the learning process. (particularly useful to
    hide the exploring) """
    mean_reward_list = []
    for i in range(num_tasks):
        # Reset what the agent has learnt for the next task
        agent.reset()
        # Used to track progress of the running
        if (i+1)%100 == 0:
            print i+1
        # Go through randomly generated n-armed bandit tasks
        reward_list = rl_cycle(agent, n_armed_bandit_tasks[i], num_steps)
        mean_reward_list.append(reward_list)

    mean_reward_list = np.mean(mean_reward_list, axis=0)
    return mean_reward_list

def main():
    # Arguments
    n = 10 # number of one-armed bandits
    num_steps = 2000 # number of times the agent pulls a lever in a task
    num_tasks = 2000 # number of random tasks to be generated
    mean_action_value = 0
    variance_action_value = 1
    variance_noise = 1

    # Generate n-armed bandit tasks
    n_armed_bandit_tasks = []
    for j in range(num_tasks):
        task = NArmedBandit(n, mean_action_value, variance_action_value,
                            variance_noise)
        n_armed_bandit_tasks.append(task)

    # Create agents
    agent0 = Agent(0, n) # Greedy
    agent01 = Agent(0.1, n)
    agent001 = Agent(0.01, n)
    agents = [agent0, agent01, agent001]

    fig = plt.figure()
    for agent in agents:
        mean_reward_list = repeat_rl_cycle(num_tasks, agent,
                                           n_armed_bandit_tasks, num_steps)
        # Plot the mean reward list
        ax = plot_reward(mean_reward_list, r'$\epsilon=%.1g$'%(agent.epsilon))

    plt.legend()
    plt.savefig('mean_reward.png')
    plt.show()

if __name__ == "__main__":
    main()
