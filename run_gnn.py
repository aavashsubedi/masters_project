import torch
from src.dataset.dataloader import get_dataloader
from src.model.model import get_model
from src.model.graph_model import get_graph_model
from src.training.trainer_graph import trainer_graph as trainer_graph
from src.dataset.warcraft_gnn_dataloader import ITRLoader
from src.training.trained_graph_debug import trainer_graph_debug

from torch_geometric.data import DataLoader
def run_gnn(cfg):

        print("Starting run.")
        root = "/share/nas2/asubedi/masters_project/data/warcraft_gnn/"
        train_dataset = ITRLoader(cfg, root, mode="train")
        val_dataset = ITRLoader(cfg, root, mode="val")
        test_dataset = ITRLoader(cfg, root, mode="test")
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)
        print("Dataset")
        model = get_graph_model(cfg)
        if cfg.gnn_debug_mode:
                trainer_graph_debug(cfg, train_dataloader, val_dataloader, test_dataloader, model)
        else:
                trainer_graph(cfg, train_dataloader, val_dataloader, test_dataloader, model)
