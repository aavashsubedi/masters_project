"""
This will be used for evaluation.
"""
import torch

def check_cost(true_weights, true_path,
    predicted_path):
    """
    We will compute the true cost of each path and then compare. if they are not equal to each other
    within a margin of error we will count thas a wrong prediction.
    We will do this using the true weights of each vertex because if the model produces the wrong weights
    then it could have a lower cost : )
    """
    #multiply each element of true_weights by the corresponding element of true_path
   # true_cost = torch.dot(true_weights, true_path)
    true_cost_b = true_weights * true_path
    true_cost_sum = true_cost_b.view(true_path.shape[0], -1).sum(dim=-1)
    predicted_cost_b = true_weights * predicted_path
    predicted_cost_sum = predicted_cost_b.view(true_path.shape[0], -1).sum(dim=-1)
   # import pdb; pdb.set_trace()
    return torch.sum(predicted_cost_sum == true_cost_sum) / true_path.shape[0]
   # predicted_cost = torch.dot(predicted_weights, true_path)
    
    #if true_cost !=

#evaluate function for evaluation/testing
import torch
import wandb
import time
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model, data_loader, criterion, 
            mode="validation"):
    start = time.time()
    #model.evaluate()
    accuracy = []
    losses = []

    for data in data_loader:
        data, label, weights = data
        data.to(device)
        label.to(device)
        weights.to(device)
        output, cnn_output = model(data)
        batchwise_accuracy = check_cost(weights, label, output)
        accuracy.append(batchwise_accuracy)
        loss = criterion(output, label)
        losses.append(loss.item())
    #import pdb; pdb.set_trace()
    avg_loss = sum(losses) / len(losses)
    avg_accuracy = sum(accuracy) / len(accuracy)
    end = time.time() - start
    results = {f"{mode}_loss": avg_loss,
               f"{mode}_accuracy": avg_accuracy,
            }
    wandb.log(results)