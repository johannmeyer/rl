import numpy as np

class OneArmedBandit():
    """ This is a class for a one-armed bandit with
    a stationary probability distribution"""

    # General Definitions
    """ Stationarity means the statistical parameters of the
    distribution are not a function of time. In this case the
    mean and variance remain constant. """

    def __init__(self, mean_action, variance_action,
                 variance_noise):
        """ Select the mean_action_value for the
        one-armed bandit from a normal distribution. """
        # Note the mean_action_value of the one-armed bandit
        # is not directly specified. Only the probability distribution
        # from which the one-armed bandit is selected is specified.
        self.mean_action_value = np.random.normal(mean_action,
                                                  np.sqrt(variance_action))

        self.std_dev_noise = np.sqrt(variance_noise)

    def get_reward(self):
        """ returns the reward of pulling
        the lever of the one-armed bandit
        with some additional noise. """
        return self.mean_action_value + np.random.normal(0, self.std_dev_noise)
