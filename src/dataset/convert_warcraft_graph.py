
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import ToSLIC
import torch
#torch transform bilinear interpolation
#torch transform resize
from torch.nn.functional import interpolate
from skimage.segmentation import slic
from skimage import graph as sk_graph
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.measure import regionprops
import networkx as nx
import skimage as sk
import torch_geometric as pyg
from tqdm import tqdm

def path_to_graph(centroids, label, sensitivity=0.25,
                  resize_shape=(96, 96)):
    """"""
    label = sk.transform.resize(label, resize_shape)
    centroid_is_path = np.zeros(len(centroids)) # One-hot encoding if a centroid is part of the path or not
    centroid = centroids.numpy()

    for row in range(len(label)):
        for column in range(len(label)):
            if label[column, row] > np.max(label) - (sensitivity*np.max(label)): # Choose the 30% brightest spots on the label only
                dists = [np.linalg.norm([row, column] - c) for c in centroid] 

                centroid_is_path[np.argmin(dists)] = 1
    
    return torch.tensor(centroid_is_path)


def path_to_graph_two(graph, weight_image):
    graph.centroid = graph.centroid/95
    graph.centroid = graph.centroid * 11
    temp = np.round(graph.centroid.numpy()).astype(int)
    values = weight_image[temp[:, 0], temp[:, 1]]
    
    #values = bilinear_interpolation(weight_image, graph.centroid)
    graph.centroid_values = torch.tensor(values)
    distances = torch.norm(graph.centroid, dim=1)
    #find the index of the maximum distance
    target = torch.argmax(distances).item()
    x = graph.centroid_values
    graph.x = x
    nx_graph = pyg.utils.to_networkx(graph)
    path = nx.shortest_path(nx_graph, source=0, target=target)

    # path is a set of integers, we want to create a tensor
    #where the path is 1 and the rest is 0 of size target
    #create a tensor of size target
    path_tensor = torch.zeros(graph.centroid.shape[0])
    #set the path nodes to 1
    path_tensor[path] = 1
    path_tensor.unsqueeze(0)
    graph.centroid_in_path = path_tensor
    return graph
def convert_image_to_graph(image, num_segments, label, compactness=10, sigma=0.0,
			   weight_image=None):
    """
    We have an image of size 12x12. We want to convert this into a graph
    where the nodes are the pixels. The edges are the pixels that are adjacent
    to each other. 
    
    The attribute of each node is a 3 dimensional vector,
    which is the RGB value of the pixel.
    """
    #first we need to create the nodes
    #the nodes are the pixels in the image
    segments = slic(image, n_segments=num_segments, compactness=compactness,
                     sigma=sigma,
                    enforce_connectivity=True)

    new_image = np.zeros((96, 96, 3))

    unique_segments = np.unique(segments)
    for segment in unique_segments:
        indices = np.where(segments == segment)
        average_colour = np.mean(image[indices], axis=0)
        new_image[indices] = average_colour
    new_image = new_image / 255

    boundaries_test = find_boundaries(segments)
    boundaries = mark_boundaries(image, segments)

    #plot boundaries and new image side by side
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(boundaries)
    # axs[1].imshow(new_image)
    # plt.show()
    rag = sk_graph.rag_mean_color(new_image, segments)
    regions = regionprops(segments)

    for region in regions:
        rag.nodes[region['label']]['centroid'] = region['centroid'][::-1]
    graph = nx.Graph(rag)

    #graph = nx.Graph(graph_two)
    #convert graph_three to pytorch geometric
    #convert the graph to a torch geometric data object
    data = pyg.utils.from_networkx(graph)
    data.pixel_count = data["pixel count"]
    data.mean_color = data["mean color"]
    data.total_color = data["total color"] 
    data.edge_attr = data.weight
    data.labels = data.labels - 1.0
    centroid_is_path = path_to_graph(data["centroid"], label)
    data["centroid_in_path_raw"] = centroid_is_path
    data = path_to_graph_two(data, weight_image)

    return data
    
    
def convert_warcraft_dataset(data_maps, data_vertex_weights, data_labels,
                              normalise=True):
    #transpose to make channel first
    #data_maps = data_maps.transpose(0, 3, 1, 2)
    # if normalise:
    #     mean = np.mean(data_maps, axis=(0, 3, 1, 2), keepdims=True)
    #     std = np.std(data_maps, axis=(0, 3, 1, 2), keepdims=True)

    #     # mean = np.mean(data_maps, axis=(0, 2, 3), keepdims=True)
    #     # std = np.std(data_maps, axis=(0, 2, 3), keepdims=True)
    #     data_maps -= mean
    #     data_maps /= std
    #     weights_max = np.max(data_vertex_weights)
    #     weights_min = np.min(data_vertex_weights)
    #     data_vertex_weights -= weights_min
    #     data_vertex_weights /= weights_max
    #apply the convert image to graph function to each image in parallel
    graph_list = []
    #use tqdm
    for i in tqdm(range(len(data_maps))):
        graph_list.append(convert_image_to_graph(data_maps[i], 200, data_labels[i],
						weight_image=data_vertex_weights[i]))
        if i > 10:
            break
    return graph_list
    # test = data_maps[0]
    # test_label = data_labels[0]
    # import pdb; pdb.set_trace()
    # data = convert_image_to_graph(test, 200, label=test_label, compactness=10, sigma=0.0)
    # import pdb; pdb.set_trace()


    return None


if __name__ == "__main__":
    convert_warcraft_dataset()
