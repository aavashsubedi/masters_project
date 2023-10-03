import torch
from src.dataset.dataloader import get_dataloader
def run(cfg):
    train_dataloader = get_dataloader(cfg, mode="train")
    for x in train_dataloader:
        data, label = x[0], x[1]
    val_dataloader = get_dataloader(cfg, mode="val")
    test_dataloader = get_dataloader(cfg, mode="test")

    import pdb; pdb.set_trace()
    pass    