import numpy as np
import torch
import gym
from torch.distributions import Normal, Categorical
import torch.nn.functional as F
from collections import deque


# 标准化动作
class NormalizedActions(gym.ActionWrapper):

    def action(self, action):       # 输入(-1, 1)的动作,转为env中原本的动作
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):       # 把env中原本的动作,转为(-1, 1)的动作
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


# 用于延迟奖励
class RewardWrapper(gym.Wrapper):
    def __init__(self, env, delay):
        super(RewardWrapper, self).__init__(env)
        self.delay = delay
        self.now_delay = 0
        self.cum_reward = np.float64(0.0)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):

        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, done), done, info

    def reward(self, reward, done):
        self.cum_reward += reward
        self.now_delay += 1
        if done:
            a = self.cum_reward
            self.cum_reward = np.float64(0.0)
            self.now_delay = 0
            return a
        elif self.now_delay >= self.delay:
            a = self.cum_reward
            self.cum_reward = np.float64(0.0)
            self.now_delay = 0
            return a
        else:
            return np.float64(0.0)


class NoisyActionWrapper(gym.Wrapper):
    def __init__(self, env, sigma):
        super(NoisyActionWrapper, self).__init__(env)
        self.sigma = sigma
        self.action_dim = env.action_space.shape[0]  # action的维度

    def step(self, action):
        noisy_action = action + np.random.normal(0, self.sigma, size=self.action_dim)
        noisy_action = noisy_action.clip(-1, 1)
        observation, reward, done, info = self.env.step(noisy_action)
        return observation, reward, done, info


class Gaussian_Mixture_Model():
    def __init__(self, logits, mu, log_sigma):
        self.logits = logits
        self.probs = logits.softmax(-1)
        self.mu = mu
        self.sigma = torch.exp(log_sigma)

        self.cat = Categorical(self.probs)
        self.batch_size = mu.size()[0]

        self.num = mu.size()[1]     # 高斯分布的数量

    def sample(self):
        idx = self.cat.sample().unsqueeze(-1)
        zz = torch.zeros(self.probs.size()).cuda()
        # print(111, self.batch_size,     # 256
        #       idx,
        #       self.mu.size())       # torch.Size([256, 3, 8])
        mask = zz.scatter_(1, idx, 1.).unsqueeze(-1).expand(self.mu.size())  # 根据idx选择
        mu = (self.mu * mask).sum(1)
        sigma = (self.sigma * mask).sum(1)
        normal = Normal(mu, sigma)
        z = normal.sample()
        return z

    def get_best(self):
        idx = torch.max(self.probs, -1, keepdim=True).indices
        zz = torch.zeros(self.probs.size()).cuda()
        mask = zz.scatter_(1, idx, 1.)  # 根据idx选择
        mask = mask.unsqueeze(-1).expand(self.mu.size())
        mu = (self.mu * mask).sum(1)
        return mu

    def rsample(self):
        mask = F.gumbel_softmax(self.logits, 1, hard=True)
        mask = mask.unsqueeze(-1).expand(self.mu.size())
        mu = (self.mu * mask).sum(1)
        sigma = (self.sigma * mask).sum(1)
        dist = Normal(mu, sigma)
        # 使用重参数化采样
        z = dist.rsample()
        action = torch.tanh(z)
        probs_entropy = self.cat.entropy().unsqueeze(-1)
        return z, action, probs_entropy

    def log_prob(self, z, action):
        batch_mu = self.mu.reshape(-1, self.mu.size()[-1])
        batch_sigma = self.sigma.reshape(-1, self.sigma.size()[-1])
        # print(111, batch_sigma.size())      # 768, 8
        # print(222, batch_mu.size())         # 768, 8
        dist = Normal(batch_mu, batch_sigma)
        # print(z.size())     # torch.Size([256, 8])
        z = z.unsqueeze(1).expand(z.size()[0], self.num, z.size()[1]).reshape(batch_sigma.size())
        action = action.unsqueeze(1).expand(action.size()[0], self.num, action.size()[1]).reshape(batch_sigma.size())

        # prob = torch.exp(log_prob)
        # prob = prob.sum(1)/self.num
        # log_prob = torch.log(prob + 1e-7)
        gaussian_prob = torch.exp(dist.log_prob(z))/((1 - action.pow(2)) + 1e-7)
        '''
        等价于
        # log_prob = dist.log_prob(z) - torch.log((1 - action.pow(2)) + 1e-7)
        # log_prob = log_prob.reshape(self.sigma.size())
        # gaussian_prob = torch.exp(log_prob)
        '''
        gaussian_prob = gaussian_prob.reshape(self.sigma.size())
        probs = self.probs.unsqueeze(-1).expand(gaussian_prob.size())
        log_prob = ((probs * gaussian_prob) + 1e-7).sum(1).log()

        log_prob = log_prob.sum(-1, keepdim=True)  # 把[256, 6]变成[256, 1]
        # TODO 这里注意看SAC原文
        # https://zhuanlan.zhihu.com/p/90671492有讲
        return log_prob


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def center_crop_image(image, output_size):      # 把图片从中间裁剪成方形
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image