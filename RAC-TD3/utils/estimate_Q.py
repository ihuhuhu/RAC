import numpy as np
import gym
import torch
from utils.others import NormalizedActions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Q_estimator:
    def __init__(self, config, agent, best_uncertain):
        self.config = config
        self.agent = agent
        self.best_uncertain = torch.FloatTensor(best_uncertain.reshape(1, -1)).to(device)

        self.eval_env = gym.make(self.config['env'])
        self.eval_env._max_episode_steps = 20000
        self.eval_env = NormalizedActions(self.eval_env)
        self.eval_env.seed(self.config['seed'] + 100)
        self.eval_env.action_space.seed(self.config['seed'])

        self.all_action = []
        num = int(100 / self.config['state_action_pairs'])
        for x in range(num, 100 + num, num):
            action_list = self.get_action_list(x, self.eval_env)
            self.all_action.append(action_list)

    def get_action_list(self, random_step, eval_env):
        action_list = []
        eval_env.seed(self.config['seed'] + 100)
        state, done = eval_env.reset(), False
        for x in range(random_step):
            action = self.agent.select_action(np.array(state),
                                              uncertain=self.best_uncertain)
            state, reward, done, _ = eval_env.step(action)
            action_list.append(action)
        return action_list

    def cal_Q_bias(self, action_list, MC_samples, max_mc_steps, eval_env):
        Q_mc = []
        for x in range(MC_samples):
            eval_env.seed(self.config['seed'] + 100)
            state, done = eval_env.reset(), False
            for action in action_list:
                last_state = state
                last_state_action = action
                state, reward, done, _ = eval_env.step(action)
            Q_mean, _ = self.agent.get_mean_std(torch.FloatTensor(last_state.reshape(1, -1)).to(device),
                                                torch.FloatTensor(last_state_action.reshape(1, -1)).to(device),
                                                self.best_uncertain)
            total_reward = reward
            for y in range(max_mc_steps):
                state_ = torch.FloatTensor(state.reshape(1, -1)).to(device)
                action = self.agent.actor(state_, self.best_uncertain*self.agent.one)
                state, reward, done, _ = eval_env.step(action.cpu().data.numpy().flatten())
                total_reward += reward * self.agent.discount ** (y + 1)
                if done:
                    break
            Q_mc.append(total_reward)
        bias = float(Q_mean) - float(np.mean(Q_mc))
        return bias, np.mean(Q_mc)

    def cal_norm_bias(self):
        bias_list = []
        Q_mc_list = []
        for actions in self.all_action:
            bias, Q_mc = self.cal_Q_bias(actions,
                                         MC_samples=self.config['MC_samples'],
                                         max_mc_steps=self.config['max_mc_steps'],
                                         eval_env=self.eval_env)
            bias_list.append(bias)
            Q_mc_list.append(Q_mc)
        norm_bias = np.array(bias_list) / abs(np.mean(Q_mc_list))
        norm_mean_bias = np.mean(norm_bias)
        norm_std_bias = np.std(norm_bias)
        return norm_mean_bias, norm_std_bias
