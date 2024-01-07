import torch
import torch.nn as nn

class HammingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None
    def forward(self, prediction, targets):
        #import pdb; pdb.set_trace()
        hamming_diff = torch.abs(targets - prediction)
        self.loss = hamming_diff.mean()
        #self.loss = hamming_diff.sum() / (hamming_diff.size(0) * targets.size(1))
        #self.loss =  self.loss.mean(dim=(1, 2))
        



        # errors = prediction * (1.0 - targets) + (1.0 - prediction) * targets
        # self.loss = errors.mean(dim=0).sum()
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
        


# loss = HammingLoss()
# prediction = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
# target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
# print(loss(prediction, target))


# class ConcreteDropout(torch.nn.module):
#     def __init(self):
#         pass