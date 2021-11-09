import numpy as np
import gym


class NormalizedActions(gym.ActionWrapper):

    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class NoisyActionWrapper(gym.Wrapper):
    def __init__(self, env, sigma):
        super(NoisyActionWrapper, self).__init__(env)
        self.sigma = sigma
        self.action_dim = env.action_space.shape[0]

    def step(self, action):
        noisy_action = action + np.random.normal(0, self.sigma, size=self.action_dim)
        noisy_action = noisy_action.clip(-1, 1)
        observation, reward, done, info = self.env.step(noisy_action)
        return observation, reward, done, info
