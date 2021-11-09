import copy
import torch
import torch.nn.functional as F
from utils.networks import Actor, Critic_Q, Temperature
from torch.distributions import Normal, Uniform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
min_Val = torch.tensor(1e-7).float()


class Agent:
    def __init__(
            self,
            state_dim,
            action_dim,
            discount=0.99,
            tau=0.005,
            policy_freq=1,
            actor_lr=3e-4,
            critic_lr=3e-4,
            temp_lr=3e-3,
            ensemble_size=10,
            batch_size=256,
            uncertain=0.5,
            explore_uncertain=0.3,
    ):
        self.actor = Actor(state_dim, action_dim).to(device).share_memory()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.last_actor = copy.deepcopy(self.actor)

        # 初始化Q值
        self.critic_Q = Critic_Q(state_dim, action_dim, ensemble_size).to(device)
        self.critic_Q_target = copy.deepcopy(self.critic_Q)
        self.critic_Q_optimizer = torch.optim.Adam(self.critic_Q.parameters(), lr=critic_lr)

        # 温度
        self.temperature = Temperature().to(device)
        self.temp_optimizer = torch.optim.Adam(self.temperature.parameters(), lr=temp_lr)
        self.target_entropy = -action_dim * 1.0

        self.total_it = 0

        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.ensemble_size = ensemble_size
        self.policy_freq = policy_freq
        self.batch_size = batch_size

        self.one = torch.ones(1, 1).to(device)
        self.uncertain_dist = Uniform(min_Val, uncertain)
        self.explore_uncertain_dist = Uniform(min_Val, explore_uncertain)

    def select_action(self, state, uncertain=None):
        if uncertain is None:
            uncertain = self.explore_uncertain_dist.sample((1, 1)).to(device)
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mu, sigma = self.actor(state, uncertain)
        dist = Normal(mu, sigma)
        z = dist.sample()
        return torch.tanh(z).cpu().data.numpy().flatten()

    def select_best_action(self, state, uncertain=0.25):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mu, log_sigma = self.actor(state, uncertain*self.one)
        return torch.tanh(mu).cpu().data.numpy().flatten()

    def get_action_log_prob(self, state, uncertain, actor=None, action=None, z=None):
        if not actor:
            actor = self.actor
        batch_mu, batch_sigma = actor(state, uncertain)
        dist = Normal(batch_mu, batch_sigma)
        if action is None:
            z = dist.rsample()
            action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log((1 - action.pow(2)) + min_Val)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, z, batch_mu, batch_sigma

    def save(self, filename):
        torch.save(self.critic_Q.state_dict(), filename + "_critic_Q_%d")
        torch.save(self.critic_Q_target.state_dict(), filename + "_critic_Q_target_%d")
        torch.save(self.critic_Q_optimizer.state_dict(), filename + "_critic_Q_optimizer_%d")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

        torch.save(self.temperature.state_dict(), filename + "_temperature")
        torch.save(self.temp_optimizer.state_dict(), filename + "_temp_optimizer")

    def load(self, filename):
        self.critic_Q.load_state_dict(torch.load(filename + "_critic_Q_%d"))
        self.critic_Q_target.load_state_dict(torch.load(filename + "_critic_Q_target_%d"))
        self.critic_Q_optimizer.load_state_dict(torch.load(filename + "_critic_Q_optimizer_%d"))

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.temperature.load_state_dict(torch.load(filename + "_temperature"))
        self.temp_optimizer.load_state_dict(torch.load(filename + "_temp_optimizer"))

    def mse_loss(self, current_value, target_value):
        return F.mse_loss(current_value, target_value, reduction='none').mean(-1).mean(-1).sum()

    def temp_loss(self, log_prob, temperature):
        return (-temperature * log_prob.detach() - temperature * self.target_entropy).mean()

    def get_mean_std(self, state, action, uncertain, target=False):
        if target:
            Q = self.critic_Q_target(state, action, uncertain)
        else:
            Q = self.critic_Q(state, action, uncertain)
        mean = Q.mean(0)
        std = Q.std(0)
        return mean, std

    def actor_loss(self, state, action, log_prob, uncertain, temperature):
        Q, _ = self.get_mean_std(state, action, uncertain)
        loss = temperature.detach() * log_prob - Q
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
            next_action, next_log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(next_state, uncertain)
            mean, std = self.get_mean_std(next_state, next_action, uncertain, target=True)
            target_Q = mean - uncertain * std
            temperature = self.temperature(uncertain)
            target_Q = reward + not_done * self.discount * (target_Q - temperature * next_log_prob)

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
            self.last_actor.load_state_dict(self.actor.state_dict())

            uncertain = self.uncertain_dist.sample((self.batch_size, 1)).to(device)
            temperature = self.temperature(uncertain)

            now_action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(state, uncertain)
            actor_loss = self.actor_loss(state, now_action, log_prob, uncertain, temperature)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            action, log_prob, z, batch_mu, batch_log_sigma = self.get_action_log_prob(state, uncertain)
            temp_loss = self.temp_loss(log_prob, temperature)
            self.temp_optimizer.zero_grad()
            temp_loss.backward()
            self.temp_optimizer.step()

            return float(actor_loss), float(temp_loss), float(-log_prob.mean())

    def cal_kl(self, samples):
        with torch.no_grad():
            state, action, next_state, reward, not_done = samples
            state = state.to(device)
            uncertain = self.uncertain_dist.sample((state.size()[0], 1)).to(device)

            now_action, log_prob, z, mu, sigma = self.get_action_log_prob(state, uncertain)

            batch_mu, batch_sigma = self.last_actor(state, uncertain)
            dist = Normal(batch_mu, batch_sigma)
            last_log_prob = dist.log_prob(z) - torch.log((1 - now_action.pow(2)) + min_Val)
            last_log_prob = last_log_prob.sum(-1, keepdim=True)
            kl = log_prob - last_log_prob

        return kl.mean()
