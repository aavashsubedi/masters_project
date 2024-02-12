import torch
from src_ska.training.train_ska import trainer_ska
from src_ska.utils.get_dist_matrix import get_dist_matrix
from torch_geometric.data import DataLoader
def run_ska(cfg):
    
    ska_graph = torch.load(cfg.ska_graph_path)
    file_path = "/share/nas2/asubedi/masters_project/data/ska/new_raw_dataset.txt"
    dist_matrix = get_dist_matrix(file_path)
    trainer_ska(cfg=cfg, ska_graph=ska_graph, dist_matrix=dist_matrix, model=None)
    print("Starting run.")
