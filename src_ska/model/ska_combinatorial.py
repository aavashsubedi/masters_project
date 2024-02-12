import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering, spectral_clustering

def k_means(data, num_clusters, max_iters=100):
    """For data we'll use eigenvalues of the Laplacian of a graph"""
    centroids = data[:num_clusters, :]
    for _ in range(max_iters):
        distances = torch.cdist(data, centroids, p=2)
        cluster_assignments = torch.argmin(distances, dim=1)
        new_centroids = torch.stack([data[cluster_assignments == i].mean(0) for i in range(num_clusters)])
        if torch.equal(new_centroids, centroids):
            break
        centroids = new_centroids
        
    return centroids, cluster_assignments
def spectral_clustering(adjacency_matrix, num_clusters=2):
    # Laplacian:
    degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=1))
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Eigenvalue Decomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix)
    eigenvectors = eigenvectors[:, 1:num_clusters+1]  # Use the first num_clusters eigenvectors
    eigenvectors = F.normalize(eigenvectors, p=2, dim=1)

    # K-Means for clustering
    centroids, cluster_assignments = k_means(eigenvectors, num_clusters)

    return cluster_assignments


def obtain_weighted_matrix(adjacency_matrix, baseline_weightings,
                           num_telescopes=197):
    
    weighted_matrix = torch.zeros((num_telescopes,
                                   num_telescopes))
    for i in range(num_telescopes):
        for j in range(num_telescopes):
            weighted_matrix[i][j] = adjacency_matrix[i][j] * (
                baseline_weightings[i] * baseline_weightings[j])
            
    return weighted_matrix


def get_results(distance_matrix, baseline_data, num_clusters=2):
    adjacency_matrix = torch.exp(-distance_matrix)
    weighted_matrix = obtain_weighted_matrix(adjacency_matrix, baseline_data)
    cluster_assignments = spectral_clustering(weighted_matrix, num_clusters)
    #get a masked version of the cluster assignments
    masked_adjacency_matrix = 
    return cluster_assignments