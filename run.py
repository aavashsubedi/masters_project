import torch
from src.dataset.dataloader import get_dataloader
from src.model.model import get_model
from src.training.trainer import trainer
from src.dataset.warcraft_gnn_dataloader import ITRLoader
def run(cfg):
#     train_dataloader = get_dataloader(cfg, mode="train")
#     val_dataloader = get_dataloader(cfg, mode="val")
#     test_dataloader = get_dataloader(cfg, mode="test")
#     model = get_model(cfg)
#     trainer(cfg, train_dataloader, val_dataloader,
#             test_dataloader, model)
        root = "/share/nas2/asubedi/masters_project/data/warcraft_gnn/"
        train_dataset = ITRLoader(cfg, root, mode="train")

        pass    