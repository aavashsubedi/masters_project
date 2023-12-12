#write a Pytorch model class 
import torch_geometric as pyg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GCNConv
from torch_geometric.nn import global_max_pool

   
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.conv1 = GCNConv(1, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.grad_approx = GradApproxGraph.apply
        self.combinatorial_solver = NewDjikstra.apply

    def forward(self, data, embedding_output=False,
                additional_feat=True):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index = edge_index.to(torch.long).to(device)
        edge_attr = edge_attr.to(torch.float).to(device)
        x = x.to(torch.float).to(device).unsqueeze(-1)
        x = self.conv1(x, edge_index, edge_attr.float())
        x = self.bn1(x)
        x = torch.nn.ReLU()(x)
        x = self.conv2(x, edge_index, edge_attr.float())
        x = self.bn2(x)
        x = torch.nn.ReLU()(x)
        x = self.conv3(x, edge_index, edge_attr.float())
        x = global_max_pool(x, data.batch)
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
    data = torch.load("/share/nas2/asubedi/masters_project/data/graph_dataset.pt")
    output = model(data)
    import pdb; pdb.set_trace()
    
class GradApproxGraph(torch.autograd.Function):
    def __init__(self, model, input_data, lambda_val=0.1,
                 example_input=None):
        """
        input_data here is a graph with the correct edges. and setup.
        Doing this will let us pass the model through the graph itself.
        """
        self.input = None
        self.output = None
        self.prev_input = example_input
        self.curr_output = None
        self.lambda_val = lambda_val
        self.model = model
        
    @staticmethod
    def forward(ctx, combinatorial_solver_output, gnn_output):
        """
        Combinatorial solver output is [num_nodes, 1]
        gnn_output is [num_nodes, 1]

        """
        ctx.save_for_backward(combinatorial_solver_output, gnn_output)
    @staticmethod
    def backward(ctx, grad_input):
        lambda_val = 0.1
        combinatorial_solver_output, gnn_output = ctx.saved_tensors
        pertubred_gnn.x = gnn_output + torch.multiply(10.0, grad_input)
        perturebed_output = NewDjikstra(pertubed_gnn)
        new_grads = (perturebed_output - combinatorial_solver_output) / 10.0
        return new_grads, new_grads
        # pertubed_gnn.x = gnn_output
        # pass 
        pass 