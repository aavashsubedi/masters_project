import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import itertools
from .graph import neighbours_8
from functools import partial
import numpy as np
from collections import namedtuple, defaultdict
import time
#add a autograd function
import torch_geometric as pyg 
import networkx as nx


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file



def Dijkstra(matrices, neighbourhood_fn=neighbours_8, request_transitions=False):
    #def forward(self, matrices, neighbourhood_fn=neighbours_8, request_transitions=False):
        # Ensure matrices is a 4D tensor (batch_size, channels, height, width)
#        matrices = torch.tensor(matrices)
        batch_size, height, width = matrices.size()
       # import pdb; pdb.set_trace()
        outputs = []
    #    start_time = time.time()
        
        #output = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])
        # if matrices.dim() != 3:
        #     raise ValueError("Input matrices must be a 3D tensor (batch_size, height, width).")


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

          #  is_unique = num_path[-1, -1] == 1

            # if request_transitions:
            #     outputs.append(output(shortest_path=on_path.unsqueeze(0), is_unique=is_unique, transitions=transitions))
            # else:
            #     outputs.append(output(shortest_path=on_path.unsqueeze(0), is_unique=is_unique, transitions=None))
            outputs.append(on_path.unsqueeze(0))
    #    shortest_paths = []
      #  mid_time = time.time() - start_time
     #   output = shortest_paths
        # for output in outputs:

        #     shortest_path = output.shortest_path
        #     #check if shortest path is just identiyty matrix
        #     # if torch.all(shortest_path == torch.eye(12)):
        #     #     print("Just identity here : (")
        #     is_unique = output.is_unique
        #     transitions = output.transitions
        #     shortest_paths.append(shortest_path)
        # final_time = time.time() - mid_time
     #   import pdb; pdb.set_trace()
        #import pdb; pdb.set_trace()
        return torch.stack(outputs).squeeze(1).requires_grad_(True)
    

# class Dijkstra(nn.Module): # Dijkstra algorithm is a combinatorial solver to find shortest paths
#     def __init__(self):
#         super(Dijkstra, self).__init__()
#         self.output = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])

#     def forward(self, matrices, neighbourhood_fn=neighbours_8, request_transitions=False):
#         # Ensure matrices is a 4D tensor (batch_size, channels, height, width)
# #        matrices = torch.tensor(matrices)
#         batch_size, height, width = matrices.size()
#         outputs = []
#         # if matrices.dim() != 3:
#         #     raise ValueError("Input matrices must be a 3D tensor (batch_size, height, width).")


#         for batch_idx in range(batch_size):
#             matrix = matrices[batch_idx]  # Extract the current matrix
#             x_max, y_max = matrix.shape
#             neighbors_func = partial(neighbourhood_fn, x_max=x_max, y_max=y_max)

#             costs = torch.full_like(matrix, 1.0e10)
#             costs[0][0] = matrix[0][0]
#             num_path = torch.zeros_like(matrix)
#             num_path[0][0] = 1
#             priority_queue = [(matrix[0][0], (0, 0))]
#             certain = set()
#             transitions = defaultdict(tuple)

#             while priority_queue:
#                 cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
#                 if (cur_x, cur_y) in certain:
#                     pass

#                 for x, y in neighbors_func(cur_x, cur_y):
#                     if (x, y) not in certain:
#                         if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
#                             costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
#                             heapq.heappush(priority_queue, (costs[x][y], (x, y)))
#                             transitions[(x, y)] = (cur_x, cur_y)
#                             num_path[x, y] = num_path[cur_x, cur_y]
#                         elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
#                             num_path[x, y] += 1

#                 certain.add((cur_x, cur_y))
            
#             # Retrieve the path
#             cur_x, cur_y = x_max - 1, y_max - 1
#             on_path = torch.zeros_like(matrix)
#             on_path[-1][-1] = 1
#             while (cur_x, cur_y) != (0, 0):
#                 cur_x, cur_y = transitions[(cur_x, cur_y)]
#                 on_path[cur_x, cur_y] = 1.0

#             is_unique = num_path[-1, -1] == 1

#             if request_transitions:
#                 outputs.append(self.output(shortest_path=on_path.unsqueeze(0), is_unique=is_unique, transitions=transitions))
#             else:
#                 outputs.append(self.output(shortest_path=on_path.unsqueeze(0), is_unique=is_unique, transitions=None))

#         shortest_paths = []
#         for output in outputs:
#             shortest_path = output.shortest_path
#             is_unique = output.is_unique
#             transitions = output.transitions
#             shortest_paths.append(shortest_path)
#         return torch.cat(shortest_paths, dim=0).requires_grad_(True)
#         return torch.cat(shortest_paths, dim=0)
#     def backward(self, grad_output):
#         #import pdb; pdb.set_trace()
#         return grad_output
    
class DijskstraClass(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        result = Dijkstra(input)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        #import pdb; pdb.set_trace()
        input_ = ctx.saved_tensors
        #why are we doing this?
       # import pdb; pdb.set_trace()
         
        return grad_output #* input_[0]
    


def DijkstraGraph(x_array, graphs):

    output = []

    # GNN weights are (nx1) tensor of ALL nodes.
    # Keep a mark of where we are in the array.
    mark_x = 0 
    for batch_idx in range(len(graphs)):
        graph = graphs[batch_idx]
        target = graph.num_nodes - 1
        x = x_array[mark_x : mark_x+graph.num_nodes]
        mark_x += graph.num_nodes
        
        graph.x = x
        nx_graph = pyg.utils.to_networkx(graph)
        path = nx.shortest_path(nx_graph, source=0, target=target)

        # path is a set of integers, we want to create a tensor
        #where the path is 1 and the rest is 0 of size target
        #create a tensor of size target
        path_tensor = torch.zeros(target+1)
        #set the path nodes to 1
        path_tensor[path] = 1
        output.append(path_tensor.unsqueeze(0))

    return torch.cat([i for i in output], dim=-1).T.requires_grad_(True)


class DijkstraGraphClass(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, graph):
        ctx.graph = graph
        result = DijkstraGraph(x, graph) # Input must be a graph
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        #input_ = ctx.saved_tensors
        #why are we doing this?
        #grad_output.shape = [123]

        return grad_output.to(device), None #* input_[0]

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

# class Dijkstra(nn.Module): # Dijkstra algorithm is a combinatorial solver to find shortest paths
#     def __init__(self):
#         super(Dijkstra, self).__init__()
#         self.output = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])

#     def forward(self, matrix, neighbourhood_fn=neighbours_8, request_transitions=False):
#         """
#         The input of this should be [x, y] and the output from cnn = [b, x, y]

#         """
#         with torch.no_grad():
#             x_max, y_max = matrix.shape
#             neighbors_func = partial((neighbourhood_fn), x_max=x_max, y_max=y_max)

#             costs = np.full_like(matrix, 1.0e10)
#             costs[0][0] = matrix[0][0]
#             num_path = np.zeros_like(matrix)
#             num_path[0][0] = 1
#             priority_queue = [(matrix[0][0], (0, 0))]
#             certain = set()
#             transitions = dict()

#             while priority_queue:
#                 cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
#                 if (cur_x, cur_y) in certain:
#                     pass

#                 for x, y in neighbors_func(cur_x, cur_y):
#                     if (x, y) not in certain:
#                         if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
#                             costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
#                             heapq.heappush(priority_queue, (costs[x][y], (x, y)))
#                             transitions[(x, y)] = (cur_x, cur_y)
#                             num_path[x, y] = num_path[cur_x, cur_y]
#                         elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
#                             num_path[x, y] += 1

#                 certain.add((cur_x, cur_y))
#             # retrieve the path
#             cur_x, cur_y = x_max - 1, y_max - 1
#             on_path = np.zeros_like(matrix)
#             on_path[-1][-1] = 1
#             while (cur_x, cur_y) != (0, 0):
#                 cur_x, cur_y = transitions[(cur_x, cur_y)]
#                 on_path[cur_x, cur_y] = 1.0

#             is_unique = num_path[-1, -1] == 1

#             if request_transitions:
#                 return self.output(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
#             else:
#                 return self.output(shortest_path=on_path, is_unique=is_unique, transitions=None)


#combinatorial_solver = Dijkstra()
#mat = np.load("sample_data.npy")[1]
#answer = np.load("sample_label.npy")
#print(combinatorial_solver(mat))
#print(answer)

