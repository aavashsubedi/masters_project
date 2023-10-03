import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

class NumpyDataset(Dataset):
    def __init__(self, data, targets=None, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if self.targets is not None:
            data = self.data[idx]
            target = self.targets[idx]
            if self.transform is not None:
                data = self.transform(data)
            #not that efficent as we are converting to tensor on the fly
            return torch.tensor(data), torch.tensor(target)
        # if self.targets is not None:
        #     return self.transform(self.data[idx]), self.targets[idx]
        # elif self.transform is not None:
        #     return self.transform(self.data[idx])
        # print("currently in the getter function")
        # return torch.tensor(self.data[idx])
        
def get_dataloader(cfg, mode="train", targets=None, transform=None):
    mode = "train" if mode == None else mode
    #assert mode in ["train", "val", "test"], "mode must be one of train, val, test"

    #assert mode in ["train", "val", "test"], "mode must be one of train, val, test"
    
    if cfg.warcraft_tile == "12":
        #we will use the 12x12 tiles
        data_path = cfg.data_dir + "/12x12/"
    elif cfg.warcraft_tile == "24":
        data_path = cfg.data_dir + "/24x24/"
    else:
        print("Choose a valid tile size")
    data_maps = np.load(data_path + mode + "_maps.npy").astype(np.float32)
    data_labels = np.load(data_path + mode + "_shortest_paths.npy").astype(np.float32)
    data_vertex_weights = np.load(data_path + mode + "_vertex_weights.npy").astype(np.float32)
    #transpose to make channel first
    #why do this? doesnt really make much sense to me :| is it to support resnet?
    data_maps = data_maps.transpose(0, 3, 1, 2)
    
    #import pdb; pdb.set_trace()
    # if cfg.normalise:
    #     #normalise the data to have zero mean and unit variance
        
        
    #     mean = np.mean(data_maps, axis=(0, 2, 3), keepdims=True)
    #     import pdb; pdb.set_trace()
        
    #     std = np.std(data_maps, axis=(0, 2, 3), keepdims=True)
        
    #     data_maps -= mean
    #     data_maps /= std
    
    #now the files are loaded, convert them to a dataiterator
    dataset = NumpyDataset(data_maps, data_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size,                             
                            shuffle=True, num_workers=cfg.num_workers)
    
    #return the dataloader
    return dataloader