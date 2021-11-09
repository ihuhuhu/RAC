import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.nn.init import calculate_gain


def kaiming_uniform_(tensor, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(-2)
    num_output_fmaps = tensor.size(-1)
    receptive_field_size = 1
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim+1, 256)
        self.l2 = nn.Linear(256, 256)

        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, state, uncertain):
        su = torch.cat([state, uncertain.log()], -1)
        a = F.relu(self.l1(su))
        a = F.relu(self.l2(a))
        mu = self.mu_head(a)

        log_std_head = self.log_std_head(a)
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        return mu, log_std_head.exp()


class Critic_Q(nn.Module):
    def __init__(self, state_dim, action_dim, ensemble_size):
        super(Critic_Q, self).__init__()
        self.l1 = EnsembleFC(state_dim + action_dim+1, 256, ensemble_size)
        self.l2 = EnsembleFC(256, 256, ensemble_size)
        self.l3 = EnsembleFC(256, 1, ensemble_size)
        self.ensemble_size = ensemble_size

    def forward(self, state, action, uncertain):
        sau = torch.cat([state, action, uncertain.log()], -1).unsqueeze(0).expand(self.ensemble_size, -1, -1)
        q1 = F.relu(self.l1(sau))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class Temperature(nn.Module):
    def __init__(self, ):
        super(Temperature, self).__init__()
        self.l1 = nn.Linear(1, 64)
        self.l2 = nn.Linear(64, 1)

    def forward(self, uncertain):
        temp = F.relu(self.l1(uncertain.log()))
        temp = self.l2(temp) - 5

        return temp.exp()
