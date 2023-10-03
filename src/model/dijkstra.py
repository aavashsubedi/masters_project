import torch
import heapq
import itertools
from functools import partial
import numpy as np
from collections import namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file
DijkstraOutput = namedtuple("DijkstraOutput", ["shortest_path", "is_unique", "transitions"])

class Graph:
    def __init__(self, vertices, matrix):
        super(Graph, self).__init__()
        self.vertices = vertices
        self.matrix = matrix
        #self.solver = shortest_distance(lambda_val=lambda_val, neighbourhood_fn=self.neighbourhood_fn)
        #self.loss_fn = hamming_loss()


    def shortest_distance(self, distance, prev_distance):
        min = 1e9
        for v in range(self.vertices):
            if distance[v] < min and prev_distance[v] == False:
                shortest = distance[v]
                shortest_index = v
 
        return shortest_index
    

    """
    def dijkstra(self, source_node=0):
        #Dijkstra's Algorithm to return an adjacency matrix. Pass in a graph object
        distance = np.full((1, self.vertices), 1e9) # Some large initialisation
        distance[source_node] = 0
        shortest_paths = np.full((1, self.vertices), False)

        for _ in range(self.vertices):
            u = self.shortest_distance(distance, shortest_paths)
            shortest_paths[u] = True # Indicating we looked at this node

            for v in range(self.vertices):
                # If not the source node or one we looked at before, and the distance is greater than the previous one:
                if (self.matrix[u][v] > 0 and sptSet[v] == False and distance[v] > distance[u] + self.matrix[u][v]):
                    distance[v] = distance[u] + self.matrix[u][v]
        
        return distance
        """
    

def hamming_loss(suggested, target):
    errors = suggested * (1.0 - target) + (1.0 - suggested) * target

    return errors.mean(dim=0).sum()


def neighbours_8(x, y, x_max, y_max):
    deltas_x = (-1, 0, 1)
    deltas_y = (-1, 0, 1)
    for (dx, dy) in itertools.product(deltas_x, deltas_y):
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def dijkstra(matrix, neighbourhood_fn=neighbours_8, request_transitions=False):
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
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=transitions)
    else:
        return DijkstraOutput(shortest_path=on_path, is_unique=is_unique, transitions=None)


mat = np.array([[1, 1, 6, 3, 7, 5, 1, 7, 4, 2],
    [1, 3, 8, 1, 3, 7, 3, 6, 7, 2],
    [2, 1, 3, 6, 5, 1, 1, 3, 2, 8],
    [3, 6, 9, 4, 9, 3, 1, 5, 6, 9],
    [7, 4, 6, 3, 4, 1, 7, 1, 1, 1],
    [1, 3, 1, 9, 1, 2, 8, 1, 3, 7],
    [1, 3, 5, 9, 9, 1, 2, 4, 2, 1],
    [3, 1, 2, 5, 4, 2, 1, 6, 3, 9],
    [1, 2, 9, 3, 1, 3, 8, 5, 2, 1],
    [2, 3, 1, 1, 9, 4, 4, 5, 8, 1]])

graph = Graph(12, mat)
print(dijkstra(graph.matrix))
