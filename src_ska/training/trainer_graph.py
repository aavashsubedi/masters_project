from src.training.optimizers import get_optimizer, get_scheulder_one_cycle, get_flat_scheduler
from src.utils.loss import HammingLoss, HammingLossGraph
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from src.utils.visualise_gradients import plot_grad_flow
import wandb
from src.model.model import GradientApproximator
from src.model.combinatorial_solvers import Dijkstra, DijskstraClass
from torchviz import make_dot
from copy import deepcopy
import networkx as nx
import torch_geometric as pyg
from src.training.evaluate_graph import check_cost_graph, evaluate_gnn
#set seed


def trainer_graph(cfg, train_dataloader, val_dataloader,
            test_dataloader, model):
    
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    optimizer = get_optimizer(cfg, model)
    #given our output is a vector of size [num_nodes, 1], and the target is the same, 
    #we can just use the HammingLOss
    criterion = HammingLossGraph()
    model.train().to(device)
    #gradient_approximater = GradientApproximator(model,
    #                input_shape=(cfg.batch_size, 12, 12))
   # dijs = DijskstraClass()
    
    if cfg.scheduler:
        scheduler = get_scheulder_one_cycle(cfg, optimizer,
         len(train_dataloader), cfg.num_epochs)
    else:
        scheduler = get_flat_scheduler(cfg, optimizer)
    early_stop_counter = 0
    curr_epoch = 0
    pbar_epochs = tqdm(range(cfg.num_epochs), desc="training",
                        leave=False)
                       
    epoch = 0
    total_accuracy = []
    evaluate_gnn(model, val_dataloader, criterion, mode="val")

    for epoch in pbar_epochs:

        pbar_data = tqdm(train_dataloader, desc=f"Epoch {epoch}",
                         leave=False)
        wandb.watch(model)
        i = 0
        for data in pbar_data:
            
            optimizer.zero_grad()
            data = data.to(device)
            label = data.centroid_in_path
            label.to(device)
            #test_data = data[0]
            #test_label = data[0].centroid_in_path
            #nx_test = pyg.utils.to_networkx(test_data)
            #nx_test.x = test_data.x
            #nx_test.edge_index = test_data.edge_index
            #nx_test.edge_attr = test_data.edge_attr
            #nx_path = nx.shortest_path(nx_test, source=0, target=test_data.target.item())

            output = model(data)
            output = output.to(device)
            """
            data.num_graphs
            label.shape = [973.]
            
            
            """
          #  import pdb; pdb.set_trace()
            loss = criterion(output.squeeze(-1), label)
            loss.backward()

            optimizer.step()
            scheduler.step()
            pbar_data.set_postfix(loss=loss.item())
            wandb.log({"loss": loss.item()})

            batchwise_accuracy = check_cost_graph(data, output)
            wandb.log({"batchwise_accuracy": batchwise_accuracy})
            total_accuracy.append(batchwise_accuracy)
            plot_grad_flow(model.named_parameters())

    
        wandb.log({"epoch_accuracy": sum(total_accuracy) / len(total_accuracy)})
            #data_copy = deepcopy(data)
        evaluate_gnn(model, val_dataloader, criterion, mode="val")
    evaluate_gnn(model, test_dataloader, criterion, mode="test")