from src.training.optimizers import get_optimizer, get_scheduler_one_cycle, get_flat_scheduler
from src.model.combinatorial_solvers import Dijkstra, DijskstraClass
from src.utils.visualise_gradients import plot_grad_flow
from src.model.model import GradientApproximator
from src.utils.loss import HammingLoss
from .evaluate import check_cost
import torch
from tqdm import tqdm
import wandb
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_func(cnn_input):

    #we need to first take a copy of this and then detach it pass it through djistktra.

    pass 


def trainer(cfg, train_dataloader, val_dataloader,
            test_dataloader, model):
    optimizer = get_optimizer(cfg, model)
    criterion = HammingLoss()
    model.train().to(device)
    gradient_approximater = GradientApproximator(model, input_shape=(cfg.batch_size, 12, 12))
    
    if cfg.scheduler:
        scheduler = get_scheduler_one_cycle(cfg, optimizer, len(train_dataloader), cfg.epochs)
    else:
        scheduler = get_flat_scheduler(cfg, optimizer)
    early_stop_counter = 0
    curr_epoch = 0
    pbar_epochs = tqdm(range(cfg.num_epochs), desc="Pretraining",
                        leave=False)
                        
    #create a MSE loss criterion 
    criterion_2 = torch.nn.MSELoss()
    
    data_copy = None
    label_copy = None
    weights_copy = None
    epoch = 0
    total_accuracy = []
    for epoch in pbar_epochs:

        pbar_data = tqdm(train_dataloader, desc=f"Epoch {epoch}",
                         leave=False)
        wandb.watch(model)

        for data in pbar_data:
            data, label, weights = data

            
            output, cnn_output = model(data)
            if epoch < 0:
                loss = criterion_2(cnn_output, weights)
                loss.backward()
            else:
                loss = criterion(output, label)
                loss.backward()
            
            batchwise_accuracy = check_cost(weights, label, output)

            optimizer.step()
            scheduler.step()
            pbar_data.set_postfix(loss=loss.item())
            plot_grad_flow(model.named_parameters())


            #uncessary at the moment
            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                                 cfg.gradient_clipping)
            wandb.log({"loss": loss.item()})
            wandb.log({"batchwise_accuracy": batchwise_accuracy})
            
            
    return None