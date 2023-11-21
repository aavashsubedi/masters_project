"""
This function will convert a CSV to a pytorch dataset.
"""
import torch
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from  torch_geometric.transforms.distance import Distance 


TXT_LOC = "/workspaces/masters_project/data/ska/raw_dataset.txt"
def generate_dataset(file_path=TXT_LOC):
    #load the dataseT
    txt_file = pd.read_csv(file_path, sep=" ", header=None)
    numpy_array = np.array(txt_file.values)
    nodes = [f"{row[1]}-{row[0]}" for row in numpy_array]
    edges = set()
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if i != j:
                edge = (i, j) if i < j else (j, i)  # Ensure a consistent order
                edges.add(edge)
    
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    pos = torch.tensor([[row[2], row[3]] for row in numpy_array], dtype=torch.float)
    


    #edges = [(i, j) for i in range(len(nodes)) for j in range(len(nodes)) if i != j]
    #edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    #pos = torch.tensor([[row[2], row[3]] for row in numpy_array], dtype=torch.float)
    #create a torch tesnor of the positions which is a random initialization betweem 0 and 1
    x = torch.tensor(np.random.rand(len(nodes)), dtype=torch.float) 
    data = Data(x=x, edge_index=edge_index)
    data.pos = pos
    #now compute the edge length 
    data = Distance(norm=False)(data)
    torch.save(data, "/workspaces/masters_project/data/graph_dataset.pt")
    
    return 0

generate_dataset()