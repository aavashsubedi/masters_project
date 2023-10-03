import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import warnings
warnings.filterwarnings("ignore")

def get_optimizer(cfg, model):
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer