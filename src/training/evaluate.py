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