import numpy as np
import torch

def get_dist_matrix(file_path):
    data = np.load(file_path)
    n = len(data)
    dist_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            x_i, y_i = data[i][-2], data[i][-1]
            x_j, y_j = data[j][-2], data[j][-1]
            euclidean_distance = (x_i - x_j)**2 + (y_i - y_j)**2
            euclidean_distance = euclidean_distance**0.5
            dist_matrix[i][j] = euclidean_distance
    dist_matrix = torch.tensor(dist_matrix)