from src.training.optimizers import get_optimizer, get_scheduler_one_cycle, get_flat_scheduler
from src.utils.loss import HammingLoss
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from src.utils.visualise_gradients import plot_grad_flow
import wandb
from src.model.model import GradientApproximator
from src.model.combinatorial_solvers import Dijkstra, DijskstraClass
#from torchviz import make_dot
from copy import deepcopy


def test_func(cnn_input):
    #we need to first take a copy of this and then detach it pass it through djistktra.
    pass 


def trainer(cfg, train_dataloader, val_dataloader,
            test_dataloader, model):
    optimizer = get_optimizer(cfg, model)
    criterion = HammingLoss()
    model.train().to(device)
    gradient_approximater = GradientApproximator(model, input_shape=(cfg.batch_size, 12, 12))
    # dijs = DijskstraClass()
    
    if cfg.scheduler:
        scheduler = get_scheduler_one_cycle(cfg, optimizer, len(train_dataloader), cfg.epochs)
    else:
        scheduler = get_flat_scheduler(cfg, optimizer)

    early_stop_counter = 0
    curr_epoch = 0
    pbar_epochs = tqdm(range(cfg.num_epochs), desc="Pretraining",
                        leave=False)
    
    data_copy = None
    label_copy = None
    for epoch in pbar_epochs:

        pbar_data = tqdm(train_dataloader, desc=f"Epoch {epoch}",
                         leave=False)
        wandb.watch(model)
        i = 0
        for data in pbar_data:
            
            data, label = data
            data.to(device)
            label.to(device)

            output = model(data)
            
            loss = criterion(output, label)
            loss.backward()

            optimizer.step()
            scheduler.step()
            pbar_data.set_postfix(loss=loss.item())
            plot_grad_flow(model.named_parameters())
            
            #uncessary at the moment
            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                                 cfg.gradient_clipping)
            wandb.log({"loss": loss.item()})

    return None
