import torch
from src.dataset.dataloader import get_dataloader
from src.model.model import get_model
from src.model.graph_model import get_graph_model
from src.training.trainer import trainer
from src.training.trainer_graph import trainer_graph
from src.dataset.warcraft_gnn_dataloader import ITRLoader
from torch_geometric.data import DataLoader

def run(cfg):
        # train_dataloader = get_dataloader(cfg, mode="train")
        # val_dataloader = get_dataloader(cfg, mode="val")
        # test_dataloader = get_dataloader(cfg, mode="test")
        # model = get_model(cfg)
        # trainer(cfg, train_dataloader, val_dataloader,
        #         test_dataloader, model)
        # pass
        root = "masters_project/data/warcraft_gnn/"
        train_dataset = ITRLoader(cfg, root, mode="train")
        val_dataset = ITRLoader(cfg, root, mode="val")
        test_dataset = ITRLoader(cfg, root, mode="test")
        model = get_graph_model(cfg)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                                 num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True,
                                 num_workers=0)
        trainer_graph(cfg, train_loader, val_loader,
                                       test_loader, model)
        # pass    