import torch
import torch.nn as nn
import itertools

class Graph:
    def __init__(self, vertices, matrix):
        super(Graph, self).__init__()
        self.vertices = vertices
        self.matrix = matrix

    def shortest_distance(self, distance, prev_distance):
        min = 1e9
        for v in range(self.vertices):
            if distance[v] < min and prev_distance[v] == False:
                shortest = distance[v]
                shortest_index = v
 
        return shortest_index

def neighbours_8(x, y, x_max, y_max):
    deltas_x = (-1, 0, 1)
    deltas_y = (-1, 0, 1)
    for (dx, dy) in itertools.product(deltas_x, deltas_y):
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new
