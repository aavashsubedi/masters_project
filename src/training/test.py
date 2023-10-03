from src.training.optimizers import get_optimizer 

def trainer(cfg, train_dataloader, val_dataloader,
            test_dataloader, model):
    optimizer = get_optimizer(cfg, model)
    