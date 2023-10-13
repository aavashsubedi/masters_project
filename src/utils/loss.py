import torch

class HammingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = None
    def forward(self, prediction, targets):
        #import pdb; pdb.set_trace()
        errors = prediction * (1.0 - targets) + (1.0 - prediction) * targets
        self.loss = errors.mean(dim=0).sum()
        return self.loss

# loss = HammingLoss()
# prediction = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
# target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
# print(loss(prediction, target))


# class ConcreteDropout(torch.nn.module):
#     def __init(self):
#         pass