import torch

class HammingLoss(nn.Module):
    def __init__(self, prediction, target):
        self.prediction = prediction
        self.target = target

    def hamming_loss(self):
        errors = self.prediction * (1.0 - self.target) + (1.0 - self.prediction) * self.target

        return errors.mean(dim=0).sum()

class ConcreteDropout(nn.module):
    def __init(self):
        pass