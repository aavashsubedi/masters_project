#write a Pytorch model class 
import torch_geometric as pyg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GCNConv
from torch_geometric.nn import global_max_pool
from .combinatorial_solvers import DijkstraGraph, DijkstraGraphClass
import copy

   
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# class DeBugModel(torch.nn.Module):
#     def __init__(self, cfg, ):
#         super(DeBugModel, self).__init__()
#      #   self.cfg = cfg
     

#         node_embed_dim = 32

#         self.gine_pred_module = torch.nn.Sequential(
#             torch.nn.Linear(1,
#                              node_embed_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(node_embed_dim, node_embed_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(node_embed_dim, node_embed_dim),
#         )
#         self.gine_mid_module = torch.nn.Sequential(
#             torch.nn.Linear(node_embed_dim, node_embed_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(node_embed_dim, node_embed_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(node_embed_dim, node_embed_dim),
#         )
#         self.gine_fin_module = torch.nn.Sequential(
#             torch.nn.Linear(node_embed_dim, node_embed_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(node_embed_dim, node_embed_dim),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(node_embed_dim, 1),
#         )

#         self.conv1 = GINEConv(self.gine_pred_module, train_eps=True,
#                                 edge_dim=1)
#         self.conv2 = GINEConv(self.gine_mid_module, train_eps=True,
#                                 edge_dim=1)
#         self.conv3 = GINEConv(self.gine_fin_module, train_eps=True,
#                                 edge_dim=1) 
        
#         self.bn1 = torch.nn.BatchNorm1d(node_embed_dim)
#         self.bn2 = torch.nn.BatchNorm1d(node_embed_dim)
#         self.bn3 = torch.nn.BatchNorm1d(1)
#         # self.embed_features = torch.nn.Embedding(10, node_embed_dim).to(device)
#         # self.linear_raw = torch.nn.Linear(12, node_embed_dim).to(device) 

        

#     def forward(self, data, embedding_output=False,
#                 additional_feat=True):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         edge_index = edge_index.to(torch.long).to(device)
#         edge_attr = edge_attr.to(torch.float).to(device)
#         x = x.to(torch.float).to(device).unsqueeze(-1) #to make it [512, 1] where 512 is the #nodes
#         import pdb; pdb.set_trace()
#         x = self.conv1(x, edge_index, edge_attr.float())
#         #add a relu
#         x = self.bn1(x)
#         x = torch.nn.ReLU()(x)
#         x = self.conv2(x, edge_index, edge_attr.float())
#         x = self.bn2(x)
#         x = torch.nn.ReLU()(x)
#         x = self.conv3(x, edge_index, edge_attr.float())
#         x = self.bn3(x)
#         import pdb; pdb.set_trace()
        
#         modified_data = data.clone()
#         modified_data.x = x
#         return modified_data
        
#         # x = global_max_pool(x, data.batch)
#         return x    

class WarCraftModel(torch.nn.Module):
    def __init__(self, cfg, ):
        super(WarCraftModel, self).__init__()
        self.cfg = cfg
        self.conv1 = GCNConv(3, 32)
        #self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(32)

        self.combinatorial_solver = DijkstraGraphClass.apply
        self.grad_approx = GradApproxGraph.apply

    def forward(self, data, embedding_output=False,
                additional_feat=True):
        x, edge_index, edge_attr = data.mean_color, data.edge_index, data.edge_attr
        edge_index = edge_index.to(torch.long).to(device)
        edge_attr = edge_attr.to(torch.float).to(device)
        x = x.to(torch.float).to(device)
        #import pdb; pdb.set_trace()
        x = self.conv1(x, edge_index, edge_attr.float())
        x = self.bn1(x)
        x = torch.nn.ReLU()(x)
        #x = self.conv2(x, edge_index, edge_attr.float())
        #x = self.bn2(x)
        #x = torch.nn.ReLU()(x)
        x = self.conv3(x, edge_index, edge_attr.float())
        #x = global_max_pool(x, data.batch) # This might not work (issues with shape?)
        #gradients have issues here. 
        #x.shape here is [123, 1]. so maybe we unsqueeze here
        # NUMBER OF NODES PER GRAPH VARIES

        combinatorial_solver_output = self.combinatorial_solver(x, data)
        x = self.grad_approx(combinatorial_solver_output, x, data)

        return x 


def get_graph_model(cfg, warcraft=True):
    model = WarCraftModel(cfg)
    """
    after this we can basically output a [512, n] vector
    where n is the number of classes we want to segment to. 
    this way we basically have class allocations. 

    How do we add the constraint that each of the models can 
    """
    return model

def test_model():
    model = get_graph_model(None)
    model.to(device)
    data = torch.load("masters_project/data/graph_dataset.pt")
    output = model(data)

    
class GradApproxGraph(torch.autograd.Function):
    def __init__(self, model, input_data, lambda_val=20,
                 example_input=None):
        """
        input_data here is a graph with the correct edges. and setup.
        Doing this will let us pass the model through the graph itself.
        """
        self.input_data = input_data # graph
        self.input = None
        self.output = None
        self.prev_input = example_input
        self.curr_output = None
        self.lambda_val = lambda_val
        self.model = model
        
    @staticmethod
    def forward(ctx, combinatorial_solver_output, x, graph_obj):
        """
        Combinatorial solver output is [num_nodes, 1]
        gnn_output is [num_nodes, 1]

        """
        ctx.save_for_backward(combinatorial_solver_output, x) 
        ctx.graph = graph_obj # save_for_backward doesn't work for non-tensors
        
        return combinatorial_solver_output

    @staticmethod
    def backward(ctx, grad_input):
        lambda_val = 20
        combinatorial_solver_output, x = ctx.saved_tensors
        graph = ctx.graph

        perturbed_gnn_weights = x + torch.multiply(lambda_val, grad_input.to(device))
        perturbed_gnn_output = DijkstraGraphClass.apply(perturbed_gnn_weights, graph)
        new_grads = - (1 / lambda_val) * (combinatorial_solver_output - perturbed_gnn_output)
        
        new_grads_2 = copy.deepcopy(new_grads).to(device)
        new_grads_2.requires_grad_(True)

        return new_grads, new_grads_2, None