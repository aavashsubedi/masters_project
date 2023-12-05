import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HammingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None

    def forward(self, prediction, targets):
        hamming_diff = torch.abs(targets - prediction) # Change this abs()
        self.loss = hamming_diff.mean()
        return self.loss


# class WarcraftGNNLoss(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss = nn.BCELoss()
#     def forward(self, prediction, targets):
#         #We have a vector of size [num_nodes, 1],
#         #target is also a vector of size [num_nodes, 1]
#         #find the binary cross entropy loss
#         #import pdb; pdb.set_trace()
