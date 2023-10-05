import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import itertools
from graph import neighbours_8, Graph
from functools import partial
import numpy as np
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file


class Dijkstra(nn.Module): # Dijkstra algorithm is a combinatorial solver to find shortest paths
    def __init__(self):
        super(Dijkstra, self).__init__()
        self.output = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])

    def forward(self, matrix, neighbourhood_fn=neighbours_8, request_transitions=False):
        with torch.no_grad():
            x_max, y_max = matrix.shape
            neighbors_func = partial((neighbourhood_fn), x_max=x_max, y_max=y_max)

            costs = np.full_like(matrix, 1.0e10)
            costs[0][0] = matrix[0][0]
            num_path = np.zeros_like(matrix)
            num_path[0][0] = 1
            priority_queue = [(matrix[0][0], (0, 0))]
            certain = set()
            transitions = dict()

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
            # retrieve the path
            cur_x, cur_y = x_max - 1, y_max - 1
            on_path = np.zeros_like(matrix)
            on_path[-1][-1] = 1
            while (cur_x, cur_y) != (0, 0):
                cur_x, cur_y = transitions[(cur_x, cur_y)]
                on_path[cur_x, cur_y] = 1.0

            is_unique = num_path[-1, -1] == 1

            if request_transitions:
                return self.output(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
            else:
                return self.output(shortest_path=on_path, is_unique=is_unique, transitions=None)


#combinatorial_solver = Dijkstra()
#mat = np.load("sample_data.npy")[1]
#answer = np.load("sample_label.npy")
#print(combinatorial_solver(mat))
#print(answer)

class NeuralNet(nn.Module):
    def __init__(self, matrix, dijkstra):
        super().__init__()
        self.matrix = matrix
        self.dijkstra = dijkstra
        super(NeuralNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            self.dijkstra(matrix)
            )

    def forward(self, x): return self.dijkstra(x)

mat = np.load("sample_data.npy")[1]
model = NeuralNet(mat, Dijkstra())
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1)

loss = criterion(predictions, self.action_data) # Compute the loss

# Backpropagation and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
