import torch
from torch.utils.data import DataLoader, Dataset

class NumpyDataset(Dataset):
    def __init__(self, data, targets=None, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.transform(self.data[idx]), self.targets[idx]
        elif self.transform is not None:
            return self.transform(self.data[idx])
        else:
            return torch.from_numpy(self.data[idx])
        
def get_dataloader(cfg, mode="train", targets=None, transform=None):
    
    if cfg.warcraft_tile == "12":
        #we will use the 12x12 tiles
        data_path = cfg.data_dir + "/12x12/"
    