import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import itertools
from .graph import neighbours_8
from functools import partial
import numpy as np
from collections import namedtuple, defaultdict
#add a autograd function


device = torch.device("gpu" if torch.cuda.is_available() else "cpu") # Put on every file

# class Dijkstra(torch.autograd.Function): # Dijkstra algorithm is a combinatorial solver to find shortest paths
#     def __init__(self):
#         #super(Dijkstra, self).__init__()
#         self.output = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])
    
#     @staticmethod
#     def forward(ctx, matrices, neighbourhood_fn=neighbours_8, request_transitions=False):
#         # Ensure matrices is a 4D tensor (batch_size, channels, height, width)
#         matrices = torch.tensor(matrices)
#         batch_size, height, width = matrices.size()
#         outputs = []


#         # for batch_idx in range(batch_size):
#         matrix = matrices[batch_idx]  # Extract the current matrix
#         x_max, y_max = matrix.shape
#         neighbors_func = partial(neighbourhood_fn, x_max=x_max, y_max=y_max)

#         costs = torch.full_like(matrix, 1.0e10)
#         costs[0][0] = matrix[0][0]
#         num_path = torch.zeros_like(matrix)
#         num_path[0][0] = 1
#         priority_queue = [(matrix[0][0], (0, 0))]
#         certain = set()
#         transitions = defaultdict(tuple)

#         while priority_queue:
#             cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
#             if (cur_x, cur_y) in certain:
#                 pass

#             for x, y in neighbors_func(cur_x, cur_y):
#                 if (x, y) not in certain:
#                     if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
#                         costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
#                         heapq.heappush(priority_queue, (costs[x][y], (x, y)))
#                         transitions[(x, y)] = (cur_x, cur_y)
#                         num_path[x, y] = num_path[cur_x, cur_y]
#                     elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
#                         num_path[x, y] += 1

#             certain.add((cur_x, cur_y))
        
#         # Retrieve the path
#         cur_x, cur_y = x_max - 1, y_max - 1
#         on_path = torch.zeros_like(matrix)
#         on_path[-1][-1] = 1
#         while (cur_x, cur_y) != (0, 0):
#             cur_x, cur_y = transitions[(cur_x, cur_y)]
#             on_path[cur_x, cur_y] = 1.0

#         is_unique = num_path[-1, -1] == 1

#         if request_transitions:
#             outputs.append(ctx.output(shortest_path=on_path.unsqueeze(0),
#                 is_unique=is_unique, transitions=transitions))
#         else:
#             outputs.append(ctx.output(shortest_path=on_path.unsqueeze(0), is_unique=is_unique, transitions=None))
#         return output.shortest_path
#         shortest_paths = []
#         for output in outputs:
#             shortest_path = output.shortest_path
#             is_unique = output.is_unique
#             transitions = output.transitions
#             shortest_paths.append(shortest_path)
#        # import pdb; pdb.set_trace()
#         return torch.cat(shortest_paths, dim=0)
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output
class Dijkstra(nn.Module): # Dijkstra algorithm is a combinatorial solver to find shortest paths
    def __init__(self):
        super(Dijkstra, self).__init__()
        self.output = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])

    def forward(self, matrices, neighbourhood_fn=neighbours_8, request_transitions=False):
        # Ensure matrices is a 4D tensor (batch_size, channels, height, width)
        matrices = torch.tensor(matrices)
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

            is_unique = num_path[-1, -1] == 1

            if request_transitions:
                outputs.append(self.output(shortest_path=on_path.unsqueeze(0), is_unique=is_unique, transitions=transitions))
            else:
                outputs.append(self.output(shortest_path=on_path.unsqueeze(0), is_unique=is_unique, transitions=None))

        shortest_paths = []
        for output in outputs:
            shortest_path = output.shortest_path
            is_unique = output.is_unique
            transitions = output.transitions
            shortest_paths.append(shortest_path)
       # import pdb; pdb.set_trace()
        return torch.cat(shortest_paths, dim=0)
    def backward(self, grad_output):
        return grad_output


        
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