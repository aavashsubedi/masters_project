import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file

import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from loss.py import HammingLoss
from combinatorial_solvers.py import Dijkstra
from math import sqrt

class CombRenset18(nn.Module):

    def __init__(self, out_features, in_channels):
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        self.pool = nn.AdaptiveMaxPool2d(output_shape)
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)


    def forward(self, x):
        x = self.resnet_model.conv1(x)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x)
        x = self.resnet_model.layer1(x)
        #x = self.resnet_model.layer2(x)
        #x = self.resnet_model.layer3(x)
        #x = self.last_conv(x)
        x = self.pool(x)
        x = x.mean(dim=1)
        return x
    
class GradientApproximator():

    def __init__(self):
        self.input = None
        self.output = None
        self.prev_input = None
        self.loss = HammingLoss()
        self.combinatorial_solver = Dijkstra()

    def compute_grads(self, input):
        pass
    



def get_model(cfg, warcraft_experiment=True):
    #if we use warcraft experiment I will set the default parameters from the paper

    if warcraft_experiment:
        resnet_module = CombRenset18(out_features=144, in_channels=96)
        return resnet_module



class CNNModel(nn.Module):
    def __init__(self, cfg):
        super(CNNModel, self).__init__()
        # Idk how to read the config file

        self.k = 12 # 12 x 12 warcraft maps
        self.input = torch.randn((self.k, self.k))
        self.output = torch.randn((self.k, self.k)) # May use orthogonal initialisation later
    
    def forward_pass(self, solver=Dijkstra()):
        input = self.input.detach().cpu().numpy()
        output = solver(self.input) # Inputs for dijkstra algo
        self.output = output
        
        return output # What is the correct form for CNN? 12 x 12 !!

    def backward_pass(self, grad, lambda_val, solver=Dijkstra()):
        input, output = self.input, self.output
        input += lambda_val * grad
        perturbed_output = solver(input)

        gradient = -(1/lambda_val) * (perturbed_output - output)

        return gradient.to(device)
