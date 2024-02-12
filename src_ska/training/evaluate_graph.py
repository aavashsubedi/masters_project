"""
This will be used for evaluation.
"""
import torch
import wandb
import time
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True

def check_cost(true_weights, true_path,
    predicted_path):
    """
    We will compute the true cost of each path and then compare. if they are not equal to each other
    within a margin of error we will count thas a wrong prediction.
    We will do this using the true weights of each vertex because if the model produces the wrong weights
    then it could have a lower cost : )
    """
    true_cost_b = true_weights * true_path
    true_cost_sum = true_cost_b.view(true_path.shape[0], -1).sum(dim=-1)
    predicted_cost_b = true_weights * predicted_path
    predicted_cost_sum = predicted_cost_b.view(true_path.shape[0], -1).sum(dim=-1)
    return torch.sum(predicted_cost_sum == true_cost_sum) / true_path.shape[0]

def check_cost_graph(graph_batch, path_taken):
    """
    Because we are using the approximation ofthe label, not only will I 
    compute the cost
    """
    correct = 0
    for i in range(graph_batch.num_graphs):
        #need to figure out how to seperate the path taken here. 
        subgraph = graph_batch[i]
        path_subset = path_taken[graph_batch.batch == i]
        cost = torch.sum(subgraph.centroid_values * path_subset.squeeze(-1))
        if cost == subgraph.cost_path:
            correct += 1
    return correct / graph_batch.num_graphs
#evaluate function for evaluation/testing


@torch.no_grad()
def evaluate_gnn(model, data_loader, criterion, 
            mode="validation"):
    accuracy = []
    losses = []

    for data in data_loader:
        # import pdb; pdb.set_trace()
        data = data.to(device)
        label = data.centroid_in_path
        label.to(device)
        output = model(data)
        output = output.to(device)
        
        
        batchwise_accuracy = check_cost_graph(data, output)
        accuracy.append(batchwise_accuracy)
        loss = criterion(output.squeeze(-1), label)
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    avg_accuracy = sum(accuracy) / len(accuracy)
    results = {f"{mode}_loss": avg_loss,
               f"{mode}_accuracy": avg_accuracy,
            }
    wandb.log(results)
    return [avg_loss, avg_accuracy]