# Balancing Value Underestimation and Overestimation with Realistic Actor-Critic
Author's PyTorch implementation of Realistic Actor-Critic(RAC) for OpenAI gym tasks.

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). If you use our code or data please cite the [paper](https://arxiv.org/abs/2110.09712).

### Usage
Experiments on single environments can be run by calling:
```
cd ./RAC-SAC
python RAC_SAC.py --env Humanoid-v3 --replay_buffer_size 300000 --seed 30 --seed_num 8
python RAC_SAC.py --env Ant-v3 --replay_buffer_size 200000 --seed 30 --seed_num 8
python RAC_SAC.py --env Hopper-v3 --replay_buffer_size 1000000 --seed 30 --seed_num 8
python RAC_SAC.py --env Walker2d-v3 --replay_buffer_size 100000 --seed 30 --seed_num 8

```
Hyper-parameters can be modified with different arguments to RAC-SAC.py or RAC-TD3.py. 

### Bibtex

```
@article{li2021balancing,
  title={Balancing Value Underestimation and Overestimationwith Realistic Actor-Critic},
  author={Li, Sicen and Wang, Gang and Tang, Qinyun and Wang, Liquan},
  journal={arXiv preprint arXiv:2110.09712},
  year={2021}
}
```

