from ray import tune
from utils.Agent import Agent
import gym
import numpy as np
import torch
import time
from torch.multiprocessing import Value, Event, Process
from utils.ExperienceReplay import ReplayBuffer
from utils.others import NormalizedActions, NoisyActionWrapper
import threading
import random
from multiprocessing import Manager
from utils.Logger import Logger, VideoRecorder
from utils.estimate_Q import Q_estimator
min_Val = torch.tensor(1e-7).float()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class learner(Agent):
    def __init__(self,
                 config,
                 flag,
                 t,
                 eval_i,
                 save_flag,
                 load_flag,
                 main_loop_flag,
                 replay_buffer,
                 Logger,
                 eval_uncertain_list,
                 kwargs,
                 process_num=4):
        self.config = config
        self.UTD = config['UTD']
        self.env_name = config['env']
        self.seed = config['seed']
        self.ensemble_size = config['ensemble_size']
        self.batch_size = config['batch_size']
        self.eval_episodes = config['eval_episodes']
        self.action_noisy_sigma = config['action_noisy_sigma']
        self.eval_uncertain_list = eval_uncertain_list
        self.critic_lr = config['critic_lr']
        self.init_critic_lr = config['init_critic_lr']
        self.target_time_steps = config['target_time_steps']
        self.start_timesteps = config['start_timesteps']

        self.flag = flag
        self.t = t
        self.eval = eval_i
        self.save_flag = save_flag
        self.load_flag = load_flag
        self.main_loop_flag = main_loop_flag

        self.replay_buffer = replay_buffer
        self.Logger = Logger
        super(learner, self).__init__(**kwargs)

        self.env = gym.make(self.env_name)
        self._max_episode_steps = self.env._max_episode_steps
        self.env = NormalizedActions(self.env)
        self.env = NoisyActionWrapper(self.env, self.action_noisy_sigma)

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.episode_reward = 0.0
        self.episode_timesteps = 0.0

        self.process_num = process_num

        torch.multiprocessing.set_start_method('fork', force=True)
        manager = Manager()
        self.reward_list = manager.list()
        self.reward_list[:] = [0 for _ in self.eval_uncertain_list]

        torch.multiprocessing.set_start_method('spawn', force=True)
        self.eval_process_flag = [Event() for x in range(0, self.process_num - 1)]
        len_uncertain = len(self.eval_uncertain_list)
        self.average_eval_num = int(len_uncertain / self.process_num)

        for x in range(0, self.process_num - 1):
            p = Process(target=self.eval_process,
                        args=(self.eval_uncertain_list[self.average_eval_num * x:self.average_eval_num * (x + 1)],
                              self.reward_list,
                              range(self.average_eval_num * x, self.average_eval_num * (x + 1)),
                              self.eval_process_flag[x]),
                        )
            p.daemon = True
            p.start()
        time.sleep(2)

    def run_step(self, ):
        set_seed_everywhere(self.seed)
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        state, done = self.env.reset(), False

        while True:
            self.flag.wait()
            if self.save_flag.is_set():
                checkpoint_dir = self.Logger.get_checkpoint_dir()
                self.save(checkpoint_dir)
                # self.recode_video(self.eval_uncertain_list, checkpoint_dir)
                self.clear_flag(self.save_flag)
                continue

            if self.load_flag.is_set():
                checkpoint_dir = self.Logger.get_checkpoint_dir()
                self.load(checkpoint_dir)
                self.clear_flag(self.load_flag)
                continue

            if self.eval.is_set():
                self.eval_policy(self.process_num)
                self.clear_flag(self.eval)
                continue

            self.episode_timesteps += 1
            if self.t.value:
                action = self.select_action(np.array(state))
            else:
                action = self.env.action_space.sample()
                action = self.env.reverse_action(action)
            next_state, reward, self.done, _ = self.env.step(action)
            self.episode_reward += reward
            done_bool = float(self.done) if self.episode_timesteps < self._max_episode_steps else 0
            self.replay_buffer.add(state, action, next_state, reward, done_bool)
            state = next_state

            self.Logger.set_timesteps()
            if self.done:
                state, done = self.env.reset(), False
                self.episode_timesteps = 0
                self.episode_reward = 0.0

            if self.t.value:
                replay_buffer_size = self.replay_buffer.get_size()
                if replay_buffer_size <= self.target_time_steps:
                    self.update_lr(replay_buffer_size)
                for x in range(self.UTD):
                    samples = self.replay_buffer.sample(self.batch_size)
                    critic_loss_list = self.train_critic(samples)
                self.Logger.store_critic_loss(critic_loss_list)

                if (self.total_it + 1) % self.policy_freq == 0:
                    samples = self.replay_buffer.sample(self.batch_size)
                    actor_loss = self.train_actor(samples)
                    self.Logger.store_actor_loss(actor_loss)
                else:
                    self.total_it += 1
            self.flag.clear()
            self.main_loop_flag.set()

    def eval_policy(self, process_num=4):
        [x.set() for x in self.eval_process_flag]
        len_uncertain = len(self.eval_uncertain_list)
        self.eval_(self.eval_uncertain_list[self.average_eval_num*(process_num-1):],
                   self.reward_list,
                   range(self.average_eval_num*(process_num-1), len_uncertain))

        for x in self.eval_process_flag:
            while x.is_set():
                time.sleep(0.02)
        reward_list = list(self.reward_list)
        self.Logger.store_result(reward_list)

        best_uncertain = self.eval_uncertain_list[reward_list.index(max(reward_list))]
        if self.config['cal_Q_error']:
            Q_estimate_class = Q_estimator(self.config, self, best_uncertain)
            mean, std = Q_estimate_class.cal_norm_bias()
            self.Logger.store_Q_error(mean, std)

        self.reward_list[:] = [0 for _ in self.eval_uncertain_list]

    def eval_process(self, uncertain_list, reward_list, num, flag):
        while True:
            flag.wait()
            self.eval_(uncertain_list, reward_list, num)
            flag.clear()

    def eval_(self, uncertain_list, reward_list, num):
        for x in range(len(uncertain_list)):
            eval_env = NormalizedActions(gym.make(self.env_name))
            eval_env.seed(self.seed + 100)
            eval_env.action_space.seed(self.seed)

            avg_reward = 0.0
            for _ in range(self.eval_episodes):
                state, done = eval_env.reset(), False
                while not done:
                    action = self.select_best_action(np.array(state),
                                                     uncertain=uncertain_list[x])
                    state, reward, done, _ = eval_env.step(action)
                    avg_reward += reward
            avg_reward /= self.eval_episodes
            reward_list[num[x]] = avg_reward

    def clear_flag(self, flag):
        flag.clear()
        self.flag.clear()
        self.main_loop_flag.set()

    def recode_video(self, uncertain_list, dir_name):
        Recorder = VideoRecorder(dir_name)
        for x in range(len(uncertain_list)):

            eval_env = NormalizedActions(gym.make(self.env_name))
            eval_env = NoisyActionWrapper(eval_env, self.action_noisy_sigma)
            eval_env.seed(self.seed + 100)

            Recorder.init()

            file_name = str(uncertain_list[x]) + '.mp4'
            state, done = eval_env.reset(), False
            while not done:
                Recorder.record(eval_env)
                action = self.select_best_action(np.array(state), uncertain=uncertain_list[x])
                state, reward, done, _ = eval_env.step(action)
            Recorder.save(file_name)

    def update_lr(self, replay_buffer_size):
        percent = np.clip((replay_buffer_size - self.start_timesteps) / (self.target_time_steps - self.start_timesteps),
                          0, 1)
        lr = self.init_critic_lr + percent * (self.critic_lr - self.init_critic_lr)
        for parm in self.critic_Q_optimizer.param_groups:
            parm['lr'] = lr


# __Trainable_class__
class RAC_TD3(tune.Trainable):

    def setup(self, config):
        self.config = config
        torch.cuda.empty_cache()
        set_seed_everywhere(config['seed'])

        env = NormalizedActions(gym.make(config['env']))
        env = NoisyActionWrapper(env, config['action_noisy_sigma'])
        env.seed(config['seed'])
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.replay_buffer = ReplayBuffer(state_dim,
                                          action_dim,
                                          max_size=config['replay_buffer_size'])
        self.Logger = Logger()
        self.Flag = Event()

        self.eval_i = Event()

        self.save_flag = Event()
        self.load_flag = Event()

        self.main_loop_flag = Event()
        self.t = Value('i', 0)
        kwargs = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'discount': self.config['discount'],
            'tau': self.config['tau'],
            'policy_noise': self.config['policy_noise'],
            'noise_clip': self.config['noise_clip'],
            'policy_freq': self.config['policy_freq'],
            'actor_lr': self.config['actor_lr'],
            'critic_lr': self.config['critic_lr'],
            'expl_noise': self.config['expl_noise'],
            'ensemble_size': self.config['ensemble_size'],
            'batch_size': self.config['batch_size'],
            'uncertain': self.config['uncertain'],
            'explore_uncertain': self.config['explore_uncertain'],
        }
        a = config['uncertain']/config['eval_uncertain_num']
        self.eval_uncertain_list = list(np.arange(a, config['uncertain']+a, a))
        Learner = learner(
            config,
            self.Flag,
            self.t,
            self.eval_i,
            self.save_flag,
            self.load_flag,
            self.main_loop_flag,
            self.replay_buffer,
            self.Logger,
            self.eval_uncertain_list,
            kwargs,
        )

        self.learner_process = threading.Thread(target=Learner.run_step, name='learner')
        self.learner_process.daemon = True
        self.learner_process.start()
        time.sleep(0.1)
        self.eval_num = 0

        self.start_timesteps = int(config['start_timesteps'])
        self.max_timesteps = int(config['max_timesteps'])
        self.eval_freq = int(config['eval_freq'])

    def step(self):
        while True:
            if self.Logger.get_timesteps() < self.start_timesteps:
                self.operate_flag(None)

            elif self.max_timesteps > self.Logger.get_timesteps() >= self.start_timesteps:
                self.t.value = 1
                if (self.Logger.get_timesteps() - self.start_timesteps) >= self.eval_num * self.eval_freq:
                    self.eval_num += 1
                    self.operate_flag(self.eval_i)
                    return self.get_dic()
                self.operate_flag(None)
            else:
                file_name = f"{self.config['policy']}_{self.config['env']}_{self.config['seed']}"
                file_name = self.config['file_name'] + f"/models/{file_name}"
                self.Logger.set_checkpoint_dir(file_name)
                self.operate_flag(self.save_flag)

                self.eval_num += 1
                self.operate_flag(self.eval_i)
                return self.get_dic()

    def cleanup(self):
         pass

    def operate_flag(self, flag):
        self.main_loop_flag.clear()
        if flag:
            flag.set()
        self.Flag.set()
        self.main_loop_flag.wait()

    def save_checkpoint(self, checkpoint_dir):
        self.Logger.set_checkpoint_dir(checkpoint_dir + '/')
        self.operate_flag(self.save_flag)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        self.Logger.set_checkpoint_dir(checkpoint_dir + '/')
        self.operate_flag(self.load_flag)

    def get_dic(self):
        actor_loss, critic_loss, \
        eval_reward, eval_times, \
        Q_mean, Q_std = self.Logger.get_loss_reward()
        return {'episode_reward_mean': max(eval_reward),
                'all_reward': eval_reward,
                'best_uncertain': self.eval_uncertain_list[eval_reward.index(max(eval_reward))],
                'timesteps': self.Logger.get_timesteps(),
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'eval_uncertain_list': self.eval_uncertain_list,
                'Q_error_mean': Q_mean,
                'Q_error_std': Q_std,
                }


