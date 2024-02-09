import math

def baseline_dists(coordinates):
    dists = []

    for i in range(len(coordinates) - 1):
        for j in range(len(coordinates) - 1):
            x_i, y_i = coordinates[i][-2], coordinates[i][-1]
            x_j, y_j = coordinates[j][-2], coordinates[j][-1]
            euclidean_distance = (x_i - x_j)**2 + (y_i - y_j)**2
            euclidean_distance = euclidean_distance**0.5
            dists.append(euclidean_distance)

    return dists