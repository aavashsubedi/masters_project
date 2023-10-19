import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file

import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from src.utils.loss import HammingLoss
from math import sqrt
from .combinatorial_solvers import Dijkstra, DijskstraClass

def glorot_initializer(shape):
    # Calculate the scale factor for Glorot initialization
    fan_in, fan_out = shape[0], shape[1]
    scale = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out), dtype=torch.float32))
    
    # Generate random values with a uniform distribution
    return torch.rand(shape, dtype=torch.float32) * 2 * scale - scale

def get_model(cfg, warcraft_experiment=True):
    #if we use warcraft experiment I will set the default parameters from the paper
    if warcraft_experiment:
        resnet_module = CombRenset18(out_features=144, in_channels=3)
        return resnet_module


class CombRenset18(nn.Module):

    def __init__(self, out_features, in_channels):
        """
        Expected shape is [b, 3, h, w]
        """
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
        del self.resnet_model.conv1
        self.resnet_model.conv1 = nn.Conv2d(in_channels,
         64, kernel_size=7, stride=2, padding=3, bias=False)
        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        self.pool = nn.AdaptiveMaxPool2d(output_shape)
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)
        self.combinatorial_solver = DijskstraClass.apply
        self.grad_approx = GradientApproximator.apply

    def forward(self, x):
        x = self.resnet_model.conv1(x) #64, 48, 48

        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x = self.resnet_model.maxpool(x) #64, 64, 24
        x = self.resnet_model.layer1(x)
        #x = self.resnet_model.layer2(x)
        #x = self.resnet_model.layer3(x)
        #x = self.last_conv(x)
        x = self.pool(x)
        x = x.mean(dim=1)
        cnn_output = x.abs()
        combinatorial_solver_output = self.combinatorial_solver(cnn_output)
        x = self.grad_approx(combinatorial_solver_output, cnn_output)
        return x #shape is 32, 12, 12
    

class GradientApproximator(torch.autograd.Function):
    def __init__(self, model, input_shape):
        self.input = None
        self.output = None
        self.prev_cnn_input = torch.rand(input_shape)

        self.curr_output = None
        self.lambda_val = 0.1
        self.model = model
        #self.criterion = HammingLoss().requires_grad_(True)
        #self.combinatorial_solver = DijskstraClass()
        self.labels = None
        self.cnn_loss = None

    @staticmethod
    def forward(ctx, combinatorial_solver_output, cnn_output):
        ctx.save_for_backward(combinatorial_solver_output, cnn_output)
        return combinatorial_solver_output
    
    @staticmethod
    def backward(ctx, grad_input): # Deviation from paper algo, calculate grad in function
        #shape of loss grad is [1, 32, 12, 12]

        lambda_val = 20
        combinatorial_solver_output, cnn_output = ctx.saved_tensors

        perturbed_cnn_weights = cnn_output + torch.multiply(lambda_val, grad_input) # Is this variable named accurately?
        perturbed_cnn_output = DijskstraClass.apply(perturbed_cnn_weights)
        new_grads = -(1 / lambda_val) * (combinatorial_solver_output - perturbed_cnn_output)
        
        return new_grads, new_grads
    