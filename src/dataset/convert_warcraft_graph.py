"""
This module converts the warcraft graph into a format that can be used by the
"""
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import ToSLIC
import torch
#torch transform bilinear interpolation
#torch transform resize
from torch.nn.functional import interpolate
def add_edges(graph_dataset):
    """
    Using the positions of the nodes, add edges to the graph
    
    """

def convert_image_to_graph(image):
    """
    We have an image of size 12x12. We want to convert this into a graph
    where the nodes are the pixels. The edges are the pixels that are adjacent
    to each other. 
    
    The attribute of each node is a 3 dimensional vector,
    which is the RGB value of the pixel.
    """
    #first we need to create the nodes
    #the nodes are the pixels in the image
    
    graph_default = ToSLIC(channel_axis=-1, n_segments=576)(torch.tensor(image))
    import pdb; pdb.set_trace()





def convert_warcraft_dataset(mode="train", normalise=True):
    mode = "train" if mode == None else mode
    data_path  = "/share/nas2/asubedi/masters_project/data/warcraft_shortest_path_oneskin/12x12/"
    data_maps = np.load(data_path + mode + "_maps.npy").astype(np.float32)
    data_labels = np.load(data_path + mode + "_shortest_paths.npy").astype(np.float32)
    data_vertex_weights = np.load(data_path + mode + "_vertex_weights.npy").astype(np.float32)
    #transpose to make channel first
    data_maps = data_maps.transpose(0, 3, 1, 2)
    
    if normalise:
        mean = np.mean(data_maps, axis=(0, 2, 3), keepdims=True)
        std = np.std(data_maps, axis=(0, 2, 3), keepdims=True)
        data_maps -= mean
        data_maps /= std
        weights_max = np.max(data_vertex_weights)
        weights_min = np.min(data_vertex_weights)
        data_vertex_weights -= weights_min
        data_vertex_weights /= weights_max
    #apply the convert image to graph function to each image in parallel
    test = data_maps[0]
    convert_image_to_graph(test)




    return None


if __name__ == "__main__":
    convert_warcraft_dataset()