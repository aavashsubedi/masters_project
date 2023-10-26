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
