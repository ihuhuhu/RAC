import copy
import torch
import torch.nn.functional as F
from utils.networks import Actor, Critic_Q
from torch.distributions import Uniform
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
min_Val = torch.tensor(1e-7).float()


class Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            actor_lr=1e-3,
            critic_lr=1e-3,
            expl_noise=0.1,
            ensemble_size=10,
            batch_size=256,
            uncertain=0.5,
            explore_uncertain=0.3
    ):
        self.actor = Actor(state_dim, action_dim).to(device).share_memory()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_Q = Critic_Q(state_dim, action_dim, ensemble_size).to(device)
        self.critic_Q_target = copy.deepcopy(self.critic_Q)
        self.critic_Q_optimizer = torch.optim.Adam(self.critic_Q.parameters(), lr=critic_lr)

        self.total_it = 0

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.ensemble_size = ensemble_size
        self.policy_freq = policy_freq
        self.batch_size = batch_size

        self.action_dim = action_dim
        self.expl_noise = expl_noise

        self.one = torch.ones(1, 1).to(device)
        self.uncertain_dist = Uniform(min_Val, uncertain)
        self.explore_uncertain_dist = Uniform(min_Val, explore_uncertain)

    def select_action(self, state, uncertain=None):
        if uncertain is None:
            uncertain = self.explore_uncertain_dist.sample((1, 1)).to(device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state, uncertain).cpu().data.numpy().flatten()
        action = action + np.random.normal(0, self.expl_noise, size=self.action_dim)
        return action.clip(-1, 1)

    def select_best_action(self, state, uncertain):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state, uncertain*self.one).cpu().data.numpy().flatten()

    def get_mean_std(self, state, action, uncertain, target=False):
        if target:
            Q = self.critic_Q_target(state, action, uncertain)
        else:
            Q = self.critic_Q(state, action, uncertain)
        mean = Q.mean(0)
        std = Q.std(0)
        return mean, std

    def save(self, filename):
        torch.save(self.critic_Q.state_dict(), filename + "_critic_Q_%d")
        torch.save(self.critic_Q_target.state_dict(), filename + "_critic_Q_target_%d")
        torch.save(self.critic_Q_optimizer.state_dict(), filename + "_critic_Q_optimizer_%d")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic_Q.load_state_dict(torch.load(filename + "_critic_Q_%d"))
        self.critic_Q_target.load_state_dict(torch.load(filename + "_critic_Q_target_%d"))
        self.critic_Q_optimizer.load_state_dict(torch.load(filename + "_critic_Q_optimizer_%d"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

    def mse_loss(self, current_value, target_value):
        return F.mse_loss(current_value, target_value, reduction='none').mean(-1).mean(-1).sum()

    def actor_loss(self, state, action, uncertain):
        Q, _ = self.get_mean_std(state, action, uncertain)
        loss = - Q
        return loss.mean()

    def update_target_Q(self):
        for param, target_param in zip(self.critic_Q.parameters(), self.critic_Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_critic(self, samples):
        state, action, next_state, reward, not_done = samples
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        reward = reward.to(device)
        not_done = not_done.to(device)

        uncertain = self.uncertain_dist.sample((self.batch_size, 1)).to(device)
        with torch.no_grad():
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                    self.actor(next_state, uncertain) + noise
            ).clamp(-1, 1)
            mean, std = self.get_mean_std(next_state, next_action, uncertain, target=True)
            target_Q = mean - uncertain * std
            target_Q = reward + not_done * self.discount * target_Q

        current_Q = self.critic_Q(state, action, uncertain)
        critic_loss = self.mse_loss(current_Q, target_Q.unsqueeze(0).expand(self.ensemble_size, -1, -1))
        self.critic_Q_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_Q_optimizer.step()
        self.update_target_Q()

        return float(critic_loss)

    def train_actor(self, samples):
        self.total_it += 1
        if self.total_it % self.policy_freq == 0:
            state, action, next_state, reward, not_done = samples
            state = state.to(device)
            uncertain = self.uncertain_dist.sample((self.batch_size, 1)).to(device)
            action = self.actor(state, uncertain)
            actor_loss = self.actor_loss(state, action, uncertain)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            return float(actor_loss)
