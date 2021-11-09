import argparse
from main import main
from ray import tune
import numpy as np
import os
import ray
import sys

# Set the recursion depth
sys.setrecursionlimit(2000)

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  #
    ray.init()

    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="all", help='OpenAI gym environment name')
    parser.add_argument("--seed", default=30, type=int, help='Sets Gym, PyTorch and Numpy seeds')
    parser.add_argument("--seed_num", default=2, type=int, help='seed numbers')
    parser.add_argument("--start_timesteps", default=5e3, type=int, help='Time steps initial random policy is used')
    parser.add_argument("--eval_freq", default=3e3, type=int, help='How often (time steps) we evaluate')
    parser.add_argument("--max_timesteps", default=1e6, type=int, help='Max time steps to run environment')
    parser.add_argument("--batch_size", default=256, type=int, help='Batch size for both actor and critic')
    parser.add_argument("--discount", default=0.99, help='Discount factor')
    parser.add_argument("--tau", default=0.005, help='Target network update rate')
    parser.add_argument("--checkpoint_freq", default=1, type=int, help='Checkpoint saving frequency')
    parser.add_argument("--policy_freq", default=1, type=int, help='Frequency of delayed policy updates')
    parser.add_argument("--actor_lr", default=3e-4, help='Actor learning rate')
    parser.add_argument("--temp_lr", default=3e-4, help='Temperature learning rate')  #
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes needed for evaluation')
    parser.add_argument("--replay_buffer_size", default=3e5, type=int, help='Replay buffer capability')

    # critic lr warm up
    parser.add_argument("--critic_lr", default=3e-4, help='critic learning rate')
    parser.add_argument("--init_critic_lr", default=3e-5, help='Initialized critic learning rate')
    parser.add_argument("--target_time_steps", type=int, default=1e4, help='Time steps to reach the maximum critic learning rate')

    #
    parser.add_argument("--ensemble_size", default=10, type=int, help='ensemble size')
    parser.add_argument("--UTD", default=20, type=int, help='update-to-data ratio')
    parser.add_argument("--uncertain", default=0.8, type=int, help='Right side of exploitation distribution')  #
    parser.add_argument("--explore_uncertain", default=0.3, help='Right side of exploration distribution')
    parser.add_argument("--eval_uncertain_num", default=12, type=int, help='Number of discrete policies for evaluation')  #

    #
    parser.add_argument("--action_noisy_sigma", default=0)

    #
    parser.add_argument("--cal_Q_error", default=False, help='Whether to estimate the Q value error')  #
    parser.add_argument('--MC_samples', type=int, default=10, help='Number of MC samples')
    parser.add_argument('--state_action_pairs', type=int, default=10, help='Number of state-action pair samples')
    parser.add_argument('--max_mc_steps', type=int, default=1200)
    parser.add_argument("--cal_KL", default=False)  #

    parser.add_argument("--cpu_per_trial", default=1, help='CPU resources used by each trail')
    parser.add_argument("--gpu_per_trial", default=1.0, help='GPU resources used by each trail')
    args = parser.parse_args()
    args.policy = 'RAC-SAC'
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./ray_results"):
        os.makedirs("./ray_results")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    args.file_name = os.getcwd()

    max_iteration = int((args.max_timesteps - args.start_timesteps) / args.eval_freq) + 2
    assert max_iteration <= 1900

    train_loop = main.RAC_SAC

    seed = [x for x in range(args.seed, args.seed + args.seed_num)]  #
    if args.env == 'all':
        env = [
            # 'Ant-v3',
            # 'HalfCheetah-v3',
            'Humanoid-v3',
            # 'Walker2d-v3',
            # 'Hopper-v3',
        ]
    else:
        env = [args.env]

    analysis = tune.run(
        train_loop,
        name=args.policy,
        # scheduler=sched,
        stop={
            "training_iteration": max_iteration
        },
        keep_checkpoints_num=int(1e5),
        resources_per_trial={
            "cpu": args.cpu_per_trial,
            "gpu": args.gpu_per_trial,
        },
        num_samples=1,  # samples的数量
        config={  # 需要传入的参数
            'policy': args.policy,
            'file_name': args.file_name,
            "env": tune.grid_search(env),
            "seed": tune.grid_search(seed),
            "start_timesteps": args.start_timesteps,
            "eval_freq": args.eval_freq,
            "max_timesteps": args.max_timesteps,
            "batch_size": args.batch_size,
            "discount": args.discount,
            "tau": args.tau,
            'policy_freq': args.policy_freq,
            "actor_lr": args.actor_lr,
            'temp_lr': args.temp_lr,
            'eval_episodes': args.eval_episodes,
            'replay_buffer_size': args.replay_buffer_size,

            "critic_lr": args.critic_lr,
            'init_critic_lr': args.init_critic_lr,
            'target_time_steps': args.target_time_steps,

            'action_noisy_sigma': args.action_noisy_sigma,

            'ensemble_size': args.ensemble_size,
            'UTD': args.UTD,
            'uncertain': args.uncertain,
            'explore_uncertain': args.explore_uncertain,
            'eval_uncertain_num': args.eval_uncertain_num,

            'cal_Q_error': args.cal_Q_error,
            'MC_samples': args.MC_samples,
            'state_action_pairs': args.state_action_pairs,
            'max_mc_steps': args.max_mc_steps,
            'cal_KL': args.cal_KL,
        },
        local_dir='./ray_results',
        checkpoint_freq=args.checkpoint_freq,
    )
