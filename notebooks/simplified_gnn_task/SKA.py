import torch
import torch_geometric as pyg 
import numpy as np
import kornia
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, to_hetero, Linear
import torch.nn.functional as F

BIG_GRAPH = torch.load("/workspaces/masters_project/data/graph_dataset.pt")
BIG_GRAPH.name = torch.arange(0, BIG_GRAPH.num_nodes)

np.random.seed(42)
def seperate_graph(graph=BIG_GRAPH):
    total_nodes_list = np.arange(0, len(graph.x), 1)

    #choose a number between 40-60
    num_nodes_used = np.random.randint(30, 50)
    num_nodes_second = 100 - num_nodes_used

    random_nodes = np.random.choice(total_nodes_list, num_nodes_used, replace=False)
    second_nodes = np.random.choice(np.setdiff1d(total_nodes_list, random_nodes), num_nodes_second, replace=False)
    random_nodes = np.sort(random_nodes)
    second_nodes = np.sort(second_nodes)

    random_nodes = torch.tensor(random_nodes, dtype=torch.long)
    second_nodes = torch.tensor(second_nodes, dtype=torch.long)
    edge_index_1, edge_attr_1 = pyg.utils.subgraph(random_nodes,
                                                    graph.edge_index, graph.edge_attr,
                                                     relabel_nodes=True)
    if (edge_index_1.max() > len(random_nodes)) or (edge_index_1.min() < 0):
        print("error")
    edge_index_2, edge_attr_2 = pyg.utils.subgraph(second_nodes,
                                                    graph.edge_index, graph.edge_attr,
                                                    relabel_nodes=True)
    if (edge_index_2.max() > len(second_nodes)) or (edge_index_2.min() < 0):
        print("error with 1")
    # #create the subgraph data
    data_1 = Data(x=graph.x[random_nodes], edge_index=edge_index_1, edge_attr=edge_attr_1)
    data_1.pos = graph.pos[random_nodes]
    data_1.name = graph.name[random_nodes]
    
    data_2 = Data(x=graph.x[second_nodes], edge_index=edge_index_2, edge_attr=edge_attr_2)
    data_2.pos = graph.pos[second_nodes]
    data_2.name = graph.name[second_nodes]
    data_1.original_graph = torch.zeros(len(graph.x), dtype=torch.long)
    data_2.original_graph = torch.ones(len(graph.x), dtype=torch.long)

    #create a new graph. object that contains the original graph and the two subgraphs
    fin_graph = Data(x = data_1.x, edge_index = data_1.edge_index, edge_attr = data_1.edge_attr, pos_one = data_1.pos,
                      name_one = data_1.name, original_graph = data_1.original_graph,
                 x_two = data_2.x, edge_index_two = data_2.edge_index, edge_attr_two = data_2.edge_attr, pos_two = data_2.pos, name_two = data_2.name, original_graph_two = data_2.original_graph)

    return fin_graph

data_list = []
for i in range(200):
    data_list.append(seperate_graph())

data_loader = pyg.loader.DataLoader(data_list, batch_size=24,shuffle=True)


class GNNModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear( hidden_channels, 1)
        self.embedding_layer = Linear(1, hidden_channels)
    def forward(self, x, edge_index):
        #embed x to hidden dim 
        x = self.embedding_layer(x.unsqueeze(-1))
        #this is [500, 16]
        #print("initial x: ", x.shape)
        #print("initial, edge_index: ", edge_index.shape)
        #print("initial_edge index max", edge_index.max())
        #initial, edge_index:  torch.Size([2, 12250])
        #initial_edge index max tensor(549)        
        x = F.relu(self.conv1(x, edge_index))
        #print("x after conv1: ", x.shape)
        x = self.conv2(x, edge_index)
        #print("x after conv2: ", x.shape)
        edge_x = x[edge_index[0]] *  x[edge_index[1]]
        #print(edge_x.shape)
        return self.lin(edge_x)
    
class DualNetwork(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.gnnone = GNNModel(hidden_channels)
        self.gnntwo = GNNModel(hidden_channels)
    def forward(self, data):
        x1, edge_index1, edge_attr1 = data.x, data.edge_index, data.edge_attr
        x2, edge_index2, edge_attr2 = data.x_two, data.edge_index_two, data.edge_attr_two
        
        output_1 = self.gnnone(x1, edge_index1)
        output_2 = self.gnntwo(x2, edge_index2)

        #now we will multiply it by the edge_attr
        output_1 = output_1 * edge_attr1
        output_2 = output_2 * edge_attr2
        
        return output_1, output_2
    
dual_network = DualNetwork(16)
kl_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
optimizer = torch.optim.Adam(dual_network.parameters(), lr=0.01)

loss_array = []
val_loss_array = []
for epoch in range(10):
  #  val_loss_array.append(evaluate_fn(dual_network, data_loader, data_loader, kl_loss, use_val_dataset=True))

    for i in data_loader:
        optimizer.zero_grad()

      #  print(i)
        model_output = dual_network(i)
        model_output_1, model_output_2 = model_output       
        max_1 = torch.max(model_output_1)
        max_2 = torch.max(model_output_2)
        max_val = torch.max(max_1, max_2)

        #model_output_1.shape = [19688, 1]
        #model_output_2.shape = [42072, 1]
        bins = torch.linspace(0, max_val.item(), 30)

        #hist1 = torch.histogram(model_output_1, bins=bins).hist.requires_grad_()
        #hist2 = torch.histogram(model_output_2, bins=bins).hist.requires_grad_()

        hist1 = kornia.enhance.histogram(model_output_1, bins=bins, bandwidth=torch.tensor(0.9))
        hist2 = kornia.enhance.histogram(model_output_2, bins=bins, bandwidth=torch.tensor(0.9))
        hist1 = torch.log(hist1)
        hist2 = torch.log(hist2)
        import pdb; pdb.set_trace()
        loss = kl_loss(hist1, hist2)
        loss_array.append(loss.item())
        optimizer.step()

#save the loss array 
np.save("/workspaces/masters_project/notebooks/simplified_gnn_task/ska_loss.npy", loss_array)