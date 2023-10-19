import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import itertools
from .graph import neighbours_8
from functools import partial
import numpy as np
from collections import namedtuple, defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def Dijkstra(matrices, neighbourhood_fn=neighbours_8, request_transitions=False):
        
        batch_size, height, width = matrices.size()
        outputs = []
        output = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])

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
                outputs.append(output(shortest_path=on_path.unsqueeze(0), is_unique=is_unique, transitions=transitions))
            else:
                outputs.append(output(shortest_path=on_path.unsqueeze(0), is_unique=is_unique, transitions=None))

        shortest_paths = []
        for output in outputs:
            shortest_path = output.shortest_path
            is_unique = output.is_unique
            transitions = output.transitions
            shortest_paths.append(shortest_path)
        return torch.cat(shortest_paths, dim=0).requires_grad_(True)
    

# Necessary for backprop    
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
    