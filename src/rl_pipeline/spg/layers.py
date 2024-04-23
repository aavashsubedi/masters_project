# Utils and functions for gumbel-sinkhorn learning
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

