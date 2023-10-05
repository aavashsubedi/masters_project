import torch

class HammingLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    #     self.prediction = prediction
    #     self.target = target

    # def hamming_loss(self):
    #     errors = self.prediction * (1.0 - self.target) + (1.0 - self.prediction) * self.target

 #       return errors.mean(dim=0).sum()
    def forward(self, prediction, targets):
        errors = prediction * (1.0 - targets) + (1.0 - prediction) * targets
        return errors.mean(dim=0).sum()

loss = HammingLoss()
prediction = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
print(loss(prediction, target))


# class ConcreteDropout(torch.nn.module):
#     def __init(self):
#         pass