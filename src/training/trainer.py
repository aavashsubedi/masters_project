from src.training.optimizers import get_optimizer, get_scheulder_one_cycle, get_flat_scheduler
from src.utils.loss import hamming_loss
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from src.utils.visualise_gradients import plot_grad_flow
import wandb


def trainer(cfg, train_dataloader, val_dataloader,
            test_dataloader, model):
    optimizer = get_optimizer(cfg, model)
    criterion = hamming_loss()
    model.train().to(device)
    
    if cfg.scheduler:
        scheduler = get_scheulder_one_cycle(cfg, optimizer, len(train_dataloader), cfg.epochs)
    else:
        scheduler = get_flat_scheduler(cfg, optimizer)
    early_stop_counter = 0
    curr_epoch = 0
    pbar_epochs = tqdm(range(cfg.num_epochs), desc="Pretraining",
                        leave=False)
    for epoch in pbar_epochs:

        pbar_data = tqdm(train_dataloader, desc=f"Epoch {epoch}",
                         leave=False)
        for data in pbar_data:
            data = data.to(device)
            output = model(data)

            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar_data.set_postfix(loss=loss.item())
            plot_grad_flow(model.named_parameters())
            
            #uncessary at the moment
            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                                 cfg.gradient_clipping)
            wandb.log(loss.item())
