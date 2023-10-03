import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file

def hamming_loss(suggested, target):
    errors = suggested * (1.0 - target) + (1.0 - suggested) * target
    return errors.mean(dim=0).sum()


class Graph():
 
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    def printSolution(self, dist):
        print("Vertex \t Distance from Source")
        for node in range(self.V):
            print(node, "\t\t", dist[node])
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minDistance(self, dist, prev_dist):
 
        # Initialize minimum distance for next node
        min = 1e7
 
        # Search not nearest vertex not in the
        # shortest path tree
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
 
        return min_index
 
    # Function that implements Dijkstra's single source
    # shortest path algorithm for a graph represented
    # using adjacency matrix representation
    def dijkstra(self, src):
 
        dist = [1e7] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minDistance(dist, sptSet)
 
            # Put the minimum distance vertex in the
            # shortest path tree
            sptSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
                if (self.graph[u][v] > 0 and
                   sptSet[v] == False and
                   dist[v] > dist[u] + self.graph[u][v]):
                    dist[v] = dist[u] + self.graph[u][v]
 
        self.printSolution(dist)






class Graph:
    def __init__(self, vertices):
        super(Graph, self).__init__()
        self.vertices = vertices
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
        distance = np.full((0, graph.vertices), 1e9) # Some large initialisation
        previous_distance = np.zeros((0, graph.vertices))





  function Dijkstra(Graph, source):
 2      
 3      for each vertex v in Graph.Vertices:
 4          dist[v] ← INFINITY
 5          prev[v] ← UNDEFINED
 6          add v to Q
 7      dist[source] ← 0
 8      
 9      while Q is not empty:
10          u ← vertex in Q with min dist[u]
11          remove u from Q
12          
13          for each neighbor v of u still in Q:
14              alt ← dist[u] + Graph.Edges(u, v)
15              if alt < dist[v]:
16                  dist[v] ← alt
17                  prev[v] ← u
18
19      return dist[], prev[]

If we are only interested in a shortest path between vertices source and target, we can terminate the search after line 10 if u = target. Now we can read the shortest path from source to target by reverse iteration:

1  S ← empty sequence
2  u ← target
3  if prev[u] is defined or u = source:          // Do something only if the vertex is reachable
4      while u is defined:                       // Construct the shortest path with a stack S
5          insert u at the beginning of S        // Push the vertex onto the stack
6          u ← prev[u]  