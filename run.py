import torch
from src.dataset.dataloader import get_dataloader
from src.model.model import get_model
from src.training.trainer import trainer

def run(cfg):
    train_dataloader = get_dataloader(cfg, mode="train")
    val_dataloader = get_dataloader(cfg, mode="val")
    test_dataloader = get_dataloader(cfg, mode="test")
    model = get_model(cfg)
    trainer(cfg, train_dataloader, val_dataloader,
            test_dataloader, model)
    

    pass    