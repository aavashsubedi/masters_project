from src.training.optimizers import get_optimizer, get_scheduler_one_cycle, get_flat_scheduler
from src.utils.loss import HammingLoss
import torch
from tqdm import tqdm
from src.utils.visualise_gradients import plot_grad_flow
import wandb
from src.model.model import GradientApproximator
from src.model.combinatorial_solvers import Dijkstra, DijskstraClass
from torchviz import make_dot
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .evaluate import check_cost, evaluate

def trainer_graph(cfg, train_dataloader, val_dataloader,
            test_dataloader, model):
    optimizer = get_optimizer(cfg, model)
    #given our output is a vector of size [num_nodes, 1], and the target is the same, 
    #we can just use the HammingLOss
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
                        
    #create a MSE loss criterion 
    criterion_2 = torch.nn.MSELoss()
    
    data_copy = None
    label_copy = None
    weights_copy = None
    epoch = 0
    total_accuracy = []
    # evaluate(model, val_dataloader, criterion, mode="val")

    for epoch in pbar_epochs:

        pbar_data = tqdm(train_dataloader, desc=f"Epoch {epoch}",
                         leave=False)
        wandb.watch(model)
        i = 0
        for data in pbar_data:
            
            # if data_copy != None:
            #     #skip the loop
            #     continue
            # if i == 0:
            #     if data_copy == None:
            #         data, label, weights = data
            #         data_copy = deepcopy(data)
            #         label_copy = deepcopy(label) 
            #         weights_copy = deepcopy(weights)
            #     data, label, weights  = data_copy, label_copy, weights_copy
            #     # import pdb; pdb.set_trace()
            #     i += 1
            # else:
            #     continue
            data = data.to(device) #label, weights = data
            label = data.centroid_in_path.to(device)
            #weights.to(device)
            
            output, graph = model(data)
            output = output.to(device)
            #import pdb; pdb.set_trace()

            loss = criterion(output, label)
            loss.backward()

            optimizer.step()
            scheduler.step()
            pbar_data.set_postfix(loss=loss.item())
           # plot_grad_flow(model.named_parameters())

            #dot = make_dot(loss, params=dict(model.named_parameters()))
            #import pdb; pdb.set_trace() 
            
            #uncessary at the moment
            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                                 cfg.gradient_clipping)
            wandb.log({"loss": loss.item()})
           # wandb.log({"batchwise_accuracy": batchwise_accuracy})
    
    #         #data_copy = deepcopy(data)
    #     evaluate(model, val_dataloader, criterion, mode="val")
    # evaluate(model, test_dataloader, criterion, mode="test")