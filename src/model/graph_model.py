#write a Pytorch model class 
import torch_geometric as pyg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.nn import global_max_pool

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DeBugModel(torch.nn.Module):
    def __init__(self, cfg, ):
        super(DeBugModel, self).__init__()
     #   self.cfg = cfg
    
        node_embed_dim = 32

        self.gine_pred_module = torch.nn.Sequential(
            torch.nn.Linear(1,
                             node_embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(node_embed_dim, node_embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(node_embed_dim, node_embed_dim),
        )
        self.gine_mid_module = torch.nn.Sequential(
            torch.nn.Linear(node_embed_dim, node_embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(node_embed_dim, node_embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(node_embed_dim, node_embed_dim),
        )
        self.gine_fin_module = torch.nn.Sequential(
            torch.nn.Linear(node_embed_dim, node_embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(node_embed_dim, node_embed_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(node_embed_dim, 1),
        )

        self.conv1 = GINEConv(self.gine_pred_module, train_eps=True,
                                edge_dim=1)
        self.conv2 = GINEConv(self.gine_mid_module, train_eps=True,
                                edge_dim=1)
        self.conv3 = GINEConv(self.gine_fin_module, train_eps=True,
                                edge_dim=1) 
        
        self.bn1 = torch.nn.BatchNorm1d(node_embed_dim)
        self.bn2 = torch.nn.BatchNorm1d(node_embed_dim)
        self.bn3 = torch.nn.BatchNorm1d(1)
        # self.embed_features = torch.nn.Embedding(10, node_embed_dim).to(device)
        # self.linear_raw = torch.nn.Linear(12, node_embed_dim).to(device) 

    def forward(self, data, embedding_output=False,
                additional_feat=True):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index = edge_index.to(torch.long).to(device)
        edge_attr = edge_attr.to(torch.float).to(device)
        x = x.to(torch.float).to(device).unsqueeze(-1) #to make it [512, 1] where 512 is the #nodes
        import pdb; pdb.set_trace()
        x = self.conv1(x, edge_index, edge_attr.float())
        #add a relu
        x = self.bn1(x)
        x = torch.nn.ReLU()(x)
        x = self.conv2(x, edge_index, edge_attr.float())
        x = self.bn2(x)
        x = torch.nn.ReLU()(x)
        x = self.conv3(x, edge_index, edge_attr.float())
        x = self.bn3(x)
        import pdb; pdb.set_trace()
        
        modified_data = data.clone()
        modified_data.x = x
        return modified_data
        
        # x = global_max_pool(x, data.batch)
        return x    
    
def get_graph_model(cfg, ):
    model = DeBugModel(cfg)
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
    
test_model()