import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR, MultiStepLR

import warnings
warnings.filterwarnings("ignore")

def get_optimizer(cfg, model):
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def get_scheulder_one_cycle(cfg, optimizer, num_steps_per_epoch,
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
def warcraft_paper_scheduler(cfg, optimizer, change_lr_epochs=[10, 20, 30],
                             gamma=0.1):
    """
    The paper reduces the leaarning rate at epochs 30 and 40 by 1/10
    what this means is. 

    if lr_0 = 10, at 30: lr_30 = 1, lr_40 = 0.1
    """
    scheduler = MultiStepLR(optimizer, milestones=change_lr_epochs, 
                            gamma=gamma)
    return scheduler