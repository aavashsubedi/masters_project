from src.training.optimizers import get_optimizer, get_scheduler_one_cycle, get_flat_scheduler
from src.utils.loss import HammingLoss
import torch
from tqdm import tqdm
from src.utils.visualise_gradients import plot_grad_flow
import wandb
from src.model.model import GradientApproximator
from src.model.combinatorial_solvers import Dijkstra, DijskstraClass
from .evaluate import check_cost, evaluate
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_func(cnn_input):
    #we need to first take a copy of this and then detach it pass it through djistktra.

    pass 


def trainer(cfg, train_dataloader, val_dataloader, test_dataloader, model):
    optimizer = get_optimizer(cfg, model)
    criterion = HammingLoss()
    model.train().to(device)
    gradient_approximater = GradientApproximator(model, input_shape=(cfg.batch_size, 12, 12))
    

    if cfg.scheduler:
        scheduler = get_scheduler_one_cycle(cfg, optimizer, len(train_dataloader), cfg.epochs)
    else:
        scheduler = get_flat_scheduler(cfg, optimizer)

    pbar_epochs = tqdm(range(cfg.num_epochs), desc="Pretraining", leave=False)
                        
    #create a MSE loss criterion 
    criterion_2 = torch.nn.MSELoss()
    
    epoch = 0
    for epoch in pbar_epochs:
        pbar_data = tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)
        wandb.watch(model)

        for data in pbar_data:
            data, label, weights = data
            data.to(device)
            label.to(device)
            weights.to(device)

            output, cnn_output = model(data) # 0.9s per step for batch=32
            output.to(device)
            cnn_output.to(device)

            if epoch < 0: # When does this happen?
                loss = criterion_2(cnn_output, weights)
                loss.backward()
            else:
                loss = criterion(output, label)
                loss.backward()
            
            batchwise_accuracy = check_cost(weights, label, output)
            
            optimizer.step()
            scheduler.step()

            pbar_data.set_postfix(loss=loss.item())
            #plot_grad_flow(model.named_parameters())

            wandb.log({"loss": loss.item()})
            wandb.log({"batchwise_accuracy": batchwise_accuracy})

        evaluate(model, val_dataloader, criterion, mode="validation")
    evaluate(model, test_dataloader, criterion, mode="test")

    return None