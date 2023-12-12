from src.training.optimizers import get_optimizer, get_scheulder_one_cycle, get_flat_scheduler
from src.utils.loss import HammingLoss
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from src.utils.visualise_gradients import plot_grad_flow
import wandb
from src.model.model import GradientApproximator
from src.model.combinatorial_solvers import Dijkstra, DijskstraClass
from torchviz import make_dot
from copy import deepcopy
#set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .evaulate import check_cost, evaluate




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
        scheduler = get_scheulder_one_cycle(cfg, optimizer, len(train_dataloader), cfg.epochs)
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
    evaluate(model, val_dataloader, criterion, mode="val")
    best_val_acc = 0

    for epoch in pbar_epochs:

        pbar_data = tqdm(train_dataloader, desc=f"Epoch {epoch}",
                         leave=False)
        wandb.watch(model)
        i = 0
        for data in pbar_data:
            
            data, label, weights = data
            data.to(device)
            label.to(device)
            weights.to(device)

            
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
    
            #data_copy = deepcopy(data)
        val_results = evaluate(model, val_dataloader, criterion, mode="val")
        curr_val_acc = val_results[-1]
        if curr_val_acc >= best_val_acc:
            best_val_acc = curr_val_acc
            temp_acc = curr_val_acc
            file_path = cfg.save_model_path + "warcraft_cnn_" + str(epoch) + "_" + str(temp_acc) + ".pt"
            best_model_weights = model.state_dict()
            best_epoch = epoch
        if curr_val_acc < best_val_acc:
            early_stop_counter += 1
            if early_stop_counter >= cfg.patience:
                print("Early Stopping")
                break
        else:
            early_stop_counter = 0 
            best_val_acc = curr_val_acc
        del val_results
    
    torch.save(best_model_weights, file_path)
    model.load_state_dict(torch.load(file_path))
    evaluate(model, test_dataloader, criterion, mode="test")
