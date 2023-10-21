from src.training.optimizers import get_optimizer, get_scheulder_one_cycle, get_flat_scheduler
from src.utils.loss import HammingLoss
import torch
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
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

from .evaulate import check_cost




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
            data, label, weights = data

            
            output, cnn_output = model(data)
            if epoch < 0:
                loss = criterion_2(cnn_output, weights)
                loss.backward()
            else:
                loss = criterion(output, label)
                loss.backward()
            
            batchwise_accuracy = check_cost(weights, label, output)
            
            #abs_output = output.abs() #not sure if this workls
            #loss = test_fn(abs_output)
            #loss = criterion(abs_output, label)

            #import pdb; pdb.set_trace()
            #output = gradient_approximater.forward(gradient_approximater,
            #                                                output, label) # Used forward instead of apply() for GPU friendliness
            #output.backward()
            #new_gradients = gradient_approximater.backward(gradient_approximater)
            #shortest_path.backward()
            #abs_output.backward(new_gradients)
            #gradient_approximater.backward(gradient_approximater)
            #this is just doing the hamming loss it doesnt do anything else.
            #loss = criterion(abs_output.detach(), label).detach()

            #simport pdb; pdb.set_trace()
            #optimizer.zero_grad()
            #loss.backward()
            optimizer.step()
            scheduler.step()
            pbar_data.set_postfix(loss=loss.item())
            plot_grad_flow(model.named_parameters())

            #dot = make_dot(loss, params=dict(model.named_parameters()))
            #import pdb; pdb.set_trace() 
            
            #uncessary at the moment
            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                                 cfg.gradient_clipping)
            wandb.log({"loss": loss.item()})
            wandb.log({"batchwise_accuracy": batchwise_accuracy})
            #data_copy = deepcopy(data)
