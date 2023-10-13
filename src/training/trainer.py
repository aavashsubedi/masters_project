from src.training.optimizers import get_optimizer, get_scheulder_one_cycle, get_flat_scheduler
from src.utils.loss import HammingLoss
import torch
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
from src.utils.visualise_gradients import plot_grad_flow
import wandb
from src.model.model import GradientApproximator
from torchviz import make_dot



def trainer(cfg, train_dataloader, val_dataloader,
            test_dataloader, model):
    optimizer = get_optimizer(cfg, model)
    criterion = HammingLoss()
    model.train().to(device)
    gradient_approximater = GradientApproximator(model, input_shape=(cfg.batch_size, 12, 12))
    
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
        wandb.watch(model)

        for data in pbar_data:
            #import pdb; pdb.set_trace()
            data, label = data
            output = model(data)
            abs_output = output.abs() #not sure if this workls
            #loss = criterion(abs_output, label)

            shortest_path = gradient_approximater.forward(gradient_approximater,
                                                           output, label) # Used forward instead of apply() for GPU friendliness
            
            #gradient_approximater.backward(gradient_approximater)
            #this is just doing the hamming loss it doesnt do anything else.
            loss = gradient_approximater.criterion(shortest_path, label)
            #simport pdb; pdb.set_trace()
            optimizer.zero_grad()
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

