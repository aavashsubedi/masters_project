import os
import torch
import hydra 
import wandb
from run import run
from src.utils.setup_wandb import setup_wandb
from omegaconf import OmegaConf
import torch.multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_default_device(device)

@hydra.main(version_base='1.3', config_path="config/",
             config_name="main.yaml")
def main(cfg):
    setup_wandb(cfg)
    run(cfg)
    
    return 0


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
#    main_path = "/workspace/storage/summer_lts4/src_new/config/main.yaml"
#    cfg = OmegaConf.load(main_path)
#    mp.spawn(main, args=(cfg,), nprocs=2)
    main()