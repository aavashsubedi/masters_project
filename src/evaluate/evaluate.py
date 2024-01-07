#evaluate function for evaluation/testing
import torch
import wandb
import time
import pandas as pd


@torch.no_grad()
def evaluate(model, data_loader, criterion, 
            mode="validation"):
    start = time.time()
    model.evaluate()