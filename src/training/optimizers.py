import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR

import warnings
warnings.filterwarnings("ignore")

def get_optimizer(cfg, model):
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def get_scheduler_one_cycle(cfg, optimizer, num_steps_per_epoch,
                            num_epochs):
    return torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                                max_lr=cfg.lr,
                                                epochs=num_epochs,
                                                steps_per_epoch=num_steps_per_epoch,
                                                pct_start=0.2,
                                                anneal_strategy="cos",
                                                div_factor=2.5,
                                                final_div_factor=20.0)

#get a linear scheduler
def get_flat_scheduler(cfg, optimizer, num_steps_per_epoch=None, num_epochs=None):
    
    step_size = 1000000
    return StepLR(optimizer, step_size=step_size)
