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
from .evaluate import check_cost, evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def trainer_graph(cfg, train_dataloader, val_dataloader,
            test_dataloader, model):
    
    #set seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    pbar_epochs = tqdm(range(cfg.num_epochs), desc="Pretraining",
                        leave=False)
                        
    #create a MSE loss criterion 
    criterion_2 = torch.nn.MSELoss()
    
    epoch = 0
    # evaluate(model, val_dataloader, criterion, mode="val")

    file_path = cfg.save_model_path + "/warcraft_gnn_" + str(epoch) + ".pt"
    for epoch in pbar_epochs:

        pbar_data = tqdm(train_dataloader, desc=f"Epoch {epoch}",
                         leave=False)
        wandb.watch(model)
        for data in pbar_data:
            #import pdb; pdb.set_trace()
            optimizer.zero_grad()
            
            data = data.to(device)
            label = data.centroid_in_path.to(device)
            
            output = model(data)
            output = output.to(device)
            
            loss = criterion(output, label)
            #import pdb; pdb.set_trace()
            loss.backward()

            batchwise_accuracy = check_cost(weights, label, output)

            optimizer.step()
            scheduler.step()
            
            #uncessary at the moment
            # torch.nn.utils.clip_grad_norm_(model.parameters(),
            #                                 cfg.gradient_clipping)
            wandb.log({"loss": loss.item()})
            wandb.log({"batchwise_accuracy": batchwise_accuracy})
    
    #         #data_copy = deepcopy(data)
        evaluate(model, val_dataloader, criterion, mode="val")

    torch.save(model.state_dict(), file_path)
    model.load_state_dict(torch.load(file_path))
    evaluate(model, test_dataloader, criterion, mode="test")