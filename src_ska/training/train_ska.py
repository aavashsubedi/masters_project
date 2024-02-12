import torch_geometric as pyg
import torch
import wandb
import networkx as nx
from src_ska.training.optimizers import get_optimizer, get_scheulder_one_cycle, get_flat_scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def trainer_ska(cfg, ska_graph, dist_matrix, model):
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    optimizer = get_optimizer(cfg, model)
    criterion = torch.nn.MSELoss() #use the MSE for now. 

    model.train().to(device) 
    scheduler = get_flat_scheduler(cfg, optimizer)
    ska_graph = ska_graph.to(device)

    for i in range(1, cfg.num_epochs + 1):
        optimizer.zero_grad()
        output = model(ska_graph, dist_matrix)
        output = output.to(device)
        edge_dist_one, edge_dist_two = output 
        loss = criterion(edge_dist_one, edge_dist_two)
        loss.backward()
        optimizer.step()
        scheduler.step()
        wandb.log({"loss": loss.item()})