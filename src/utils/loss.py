import torch

class HammingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, prediction, targets):
        prediction.requires_grad = True
        targets.requires_grad = True

        errors = prediction * (1.0 - targets) + (1.0 - prediction) * targets
        return errors.mean(dim=0).sum()

# loss = HammingLoss()
# prediction = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
# target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
# print(loss(prediction, target))


# class ConcreteDropout(torch.nn.module):
#     def __init(self):
#         pass