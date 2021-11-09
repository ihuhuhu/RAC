import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        max_size = int(max_size)
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = torch.zeros(max_size, state_dim)
        self.action = torch.zeros(max_size, action_dim)
        self.next_state = torch.zeros(max_size, state_dim)
        self.reward = torch.zeros(max_size, 1)
        self.not_done = torch.zeros(max_size, 1)

    def sample(self, batch_size, size_range=None):
        if not size_range:
            low = 0
        else:
            low = int(max(self.size - size_range, 0))
        ind = np.random.randint(low, self.size, size=batch_size)
        samples = self.sample_(ind)
        return samples

    def get_size(self):
        return self.size

    def add(self, state, action, next_state, reward, done):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([float(reward)])
        not_done = torch.FloatTensor([1. - done])
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = not_done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_(self, ind):
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )
