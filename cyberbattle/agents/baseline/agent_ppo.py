from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torch.cuda

device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()

class ActorCritic(nn.Module):
    pass

class PPO(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
