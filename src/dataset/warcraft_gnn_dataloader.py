import copy
import sys
import random
import os
import math
import time
from typing import List, Tuple, Union
import torch
import torch_geometric as pyg
import numpy as np
from torch_geometric.data import Data, DataLoader, Dataset, InMemoryDataset
from torch_geometric.utils import to_networkx, from_networkx
import itertools
from src.dataset.convert_warcraft_graph import convert_image_to_graph, convert_warcraft_dataset

class ITRLoader(InMemoryDataset):
    
    def __init__(self, cfg, root, transform=None, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pre_path = ""
        super(ITRLoader, self).__init__(root, transform, None)
        self.data, self.slices = torch.load(self.processed_paths[-1],
                                            map_location=self.device)
    
    @property
    def processed_paths(self):
        pre_path = self.cfg.processed_warcraft_dir
        if self.mode == "val":
            save_path = pre_path + "/val.pt"
        elif self.mode == "test":
            save_path = pre_path + "/test.pt"
        else:
            save_path = pre_path + "/train.pt"
        return [save_path]
    
    @property 
    def raw_file_names(self) -> str:
        return []
    
    def download(self):
        pass
    @property
    def processed_file_names(self):
        if self.mode == "val":
            pass
        elif self.mode == "test":
            pass
        else:
            pass
        return 0

    def process(self):
        """
        Reads the image data and converts it into a graph
        """
        
        #go through every file in the directory and process them
        if self.mode == "train":
            data_path = self.cfg.data_dir + "/12x12/"
            data_maps = np.load(data_path + self.mode + "_maps.npy").astype(np.float32)
            data_labels = np.load(data_path + self.mode + "_shortest_paths.npy").astype(np.float32)
            data_vertex_weights = np.load(data_path + self.mode + "_vertex_weights.npy").astype(np.float32)
            graph_list = convert_warcraft_dataset(data_maps, 
                                                data_labels, 
                                                data_vertex_weights)      
        import pdb; pdb.set_trace()
        test_path = "/share/nas2/asubedi/masters_project/data/warcraft_gnn/processed/"
        #data_list = list(itertools.chain.from_iterable(graph_list))
        data, slices = self.collate(graph_list)
        if self.mode == "train":
            torch.save((data, slices), self.pre_path + "train.pt")
        elif self.mode == "val":
            torch.save((data, slices), self.pre_path + "val.pt")
        else:
            torch.save((data, slices), self.pre_path + "test.pt")

    def get(self, idx:int):
        data = super().get(idx)
        return data 
