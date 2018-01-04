from one_armed_bandit import *
class NArmedBandit:

    def __init__(self, n, mean_action_value, variance_action_value, variance_noise):
        self.list_of_bandits = []
        for i in range(n):
            one_armed_bandit = OneArmedBandit(mean_action_value,
                                              variance_action_value,
                                              variance_noise)
            self.list_of_bandits.append(one_armed_bandit)

    def get_reward(self, action):
        return self.list_of_bandits[action].get_reward()
