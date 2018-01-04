import numpy as np

class Agent():
    """ This is the reinforcement learning agent that will
    try to maximise the winnings on the n-armed bandits. """

    def __init__(self, epsilon=0.1, n=10):
        self.epsilon = epsilon # The probability of exploring
        self.n = n # Number of n-armed bandits
        self.reset()

    def reset(self):
        """ Resets the agent to its initial state. """
        # Q-values
        self.action_values = np.zeros(self.n)
        # stores the number of times each action has been performed
        self.Ka = np.zeros(self.n)

    def get_action(self):
        """ Returns the index of the one-armed bandit. """
        probability = np.random.random_sample()
        if probability < self.epsilon:
            action = self.explore()
        else:
            action = self.exploit()

        # increment the number of times this action has been selected
        self.Ka[action] += 1
        return action

    def explore(self):
        """ Pick a random one-armed bandit from the list
        of n-armed bandits. """
        return np.random.random_integers(0,self.n-1)

    def exploit(self):
        """ Pick the one-armed bandit form the list of n-armed
        bandits that maximises the Q value (in other words the
        one-armed that pays out the most as far as the agent knows). """
        return np.argmax(self.action_values)

    def update_q_value(self, one_armed_bandit_idx, reward,
                       step_size=0.1):
        """ This is the incremental approach to updating the Q-value
        for a specific one-armed bandit. step_size can either diminish
        with time or remain constant. If it diminishes with time later
        updates carry less weight and will have a smaller impact on the
        Q-values. If the step_size is a constant value, the agent will
        be able to adapt to a changing environment (i.e. non-stationary)"""
        old_estimate = self.action_values[one_armed_bandit_idx]
        new_estimate = old_estimate + step_size * (reward - old_estimate)

        self.action_values[one_armed_bandit_idx] = new_estimate
