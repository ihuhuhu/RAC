import imageio
import os


class Logger:
    def __init__(self):
        self.timesteps = 0

        self.eval_reward = [0.0]
        self.eval_times = 0
        self.actor_loss = 0.0
        self.critic_loss = 0.0
        self.temp_loss = 0.0
        self.temperature = 0.0
        self.entropy = 0.0
        self.Q_std = 0.
        self.Q_mean = 0.
        self.kl = 0.

        self.checkpoint_dir = None

    def store_result(self, reward):
        self.eval_reward = reward
        self.eval_times += 1
        return

    def get_timesteps(self):
        return self.timesteps

    def set_timesteps(self):
        self.timesteps += 1
        return

    def store_actor_loss(self, actor_loss, temp_loss, entropy):
        self.actor_loss = actor_loss
        self.temp_loss = temp_loss
        self.entropy = entropy

    def store_critic_loss(self, critic_loss):
        self.critic_loss = critic_loss

    def store_temperature(self, temperature):
        self.temperature = temperature

    def get_loss_reward(self):
        return self.actor_loss, self.critic_loss, self.temp_loss, \
               self.temperature, self.eval_reward, self.eval_times, \
               self.entropy, self.Q_mean, self.Q_std, self.kl

    def store_Q_error(self, mean, std):
        self.Q_mean = mean
        self.Q_std = std

    def store_kl(self, kl):
        self.kl = kl

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def get_checkpoint_dir(self):
        return self.checkpoint_dir


class VideoRecorder(object):
    def __init__(self, dir_name, height=1200, width=1600, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            try:
                frame = env.render(
                    mode='rgb_array',
                    height=self.height,
                    width=self.width,
                    camera_id=self.camera_id
                )
            except:
                frame = env.render(
                    mode='rgb_array',
                )

            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)

