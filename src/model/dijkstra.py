import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file

def hamming_loss(suggested, target):
    errors = suggested * (1.0 - target) + (1.0 - suggested) * target

    return errors.mean(dim=0).sum()


class Graph:
    def __init__(self, vertices):
        super(Graph, self).__init__()
        self.vertices = vertices
        self.matrix = np.zeros((len(self.vertices), len(self.vertices)))
        self.solver = ShortestPath(lambda_val=lambda_val, neighbourhood_fn=self.neighbourhood_fn)
        self.loss_fn = HammingLoss()


    def shortest_distance(self, distance, prev_distance):
        min = 1e9
        for v in range(self.vertices):
            if distance[v] < min and prev_distance[v] == False:
                shortest = distance[v]
                shortest_index = v
 
        return shortest_index


    def dijkstra(graph, source_node):
        """
        Dijkstra's Algorithm to return an adjacency matrix. Pass in a graph object
        """
        distance = np.full((0, graph.vertices), 1e9) # Some large initialisation
        shortest_paths = numpy.full((0, graph.vertices), False)

        for _ in range(graph.vertices):
            u = shortest_distance(distance, shortest_paths)
            shortest_paths[u] = True # Indicating we looked at this node

            for v in range(graph.vertices):
                # If not the source node or one we looked at before, and the distance is greater than the previous one:
                if (graph[u][v] > 0 and sptSet[v] == False and distance[v] > distance[u] + graph.matrix[u][v]):
                    distance[v] = distance[u] + graph[u][v]



