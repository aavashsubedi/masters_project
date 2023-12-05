import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import heapq
import itertools
from .graph import neighbours_8, k_means
from functools import partial
import numpy as np
from collections import namedtuple, defaultdict
import time
import torch_geometric as pyg 
import networkx as nx


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file


def Dijkstra(matrices, neighbourhood_fn=neighbours_8, request_transitions=False): # Shortest path finder
        batch_size, height, width = matrices.size()
        outputs = []

        for batch_idx in range(batch_size):
            matrix = matrices[batch_idx]  # Extract the current matrix
            x_max, y_max = matrix.shape
            neighbors_func = partial(neighbourhood_fn, x_max=x_max, y_max=y_max)

            costs = torch.full_like(matrix, 1.0e10)
            costs[0][0] = matrix[0][0]
            num_path = torch.zeros_like(matrix)
            num_path[0][0] = 1
            priority_queue = [(matrix[0][0], (0, 0))]
            certain = set()
            transitions = defaultdict(tuple)

            while priority_queue:
                cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
                if (cur_x, cur_y) in certain:
                    pass

                for x, y in neighbors_func(cur_x, cur_y):
                    if (x, y) not in certain:
                        if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                            costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                            heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                            transitions[(x, y)] = (cur_x, cur_y)
                            num_path[x, y] = num_path[cur_x, cur_y]
                        elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                            num_path[x, y] += 1

                certain.add((cur_x, cur_y))
            
            # Retrieve the path
            cur_x, cur_y = x_max - 1, y_max - 1
            on_path = torch.zeros_like(matrix)
            on_path[-1][-1] = 1
            while (cur_x, cur_y) != (0, 0):
                cur_x, cur_y = transitions[(cur_x, cur_y)]
                on_path[cur_x, cur_y] = 1.0

            outputs.append(on_path.unsqueeze(0))

        return torch.stack(outputs).squeeze(1).requires_grad_(True)
    
    
class DijskstraClass(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        result = Dijkstra(input)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input_ = ctx.saved_tensors
        #why are we doing this?
         
        return grad_output #* input_[0]
    
####################################################### Dijkstra on graphs
def DijkstraGraph(graph):
    nx_graph = pyg.utils.to_networkx(graph)
    path = nx.shortest_path(nx_graph, source=0, target=122)

    return torch.tensor(path)


class DijskstraGraphClass(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        result = DijkstraGraph(input) # Input must be a graph
        return result
    @staticmethod
    def backward(ctx, grad_output):
        input_ = ctx.saved_tensors
        #why are we doing this?
         
        return grad_output #* input_[0]

####################################################### Spectral Clustering on graphs

def spectral_clustering(adjacency_matrix, num_clusters=2):
    # Laplacian:
    degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Eigenvalue Decomposition
    eigenvalues, eigenvectors = torch.symeig(laplacian_matrix, eigenvectors=True)
    eigenvectors = eigenvectors[:, 1:num_clusters+1]  # Use the first num_clusters eigenvectors
    eigenvectors = F.normalize(eigenvectors, p=2, dim=1)

    # K-Means for clustering
    centroids, cluster_assignments = k_means(eigenvectors, num_clusters)

    return cluster_assignments
