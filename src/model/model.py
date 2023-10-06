import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file

import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from .loss import HammingLoss
from math import sqrt
from combinatorial_solvers import Dijkstra

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

    def __init__(self, model):
        self.input = None
        self.output = None
        self.prev_cnn_input = None
        self.curr_output = None
        self.lambda_val = 0.1
        self.model = model
        self.loss = HammingLoss()
        self.combinatorial_solver = Dijkstra()
        self.labels = None
        self.cnn_loss = None

    def forward_pass(self, input):
        if self.prev_cnn_input == None:
            #take the shape of the input and create a tensor of random numbers with the same shape
            self.prev_cnn_input = torch.rand(input.shape)

        self.cnn_input = input.detach().cpu().numpy()
        self.combinatorial_output = self.combinatorial_solver(self.cnn_input)
        self.prev_cnn_input = self.output

    def backward_pass(self):
        #input to the backward pass is from the forward pass before djikstra and after djikstra
        loss = self.loss(self.combinatorial_output, self.labels)
        loss_grad = loss.detach().grad

        perturbed_cnn_weights   = self.prev_cnn_input + self.lambda_val * loss_grad
        perturbed_cnn_output    = self.combinatorial_solver(perturbed_cnn_weights)
        new_grads = -(1/self.lambda_val) * (perturbed_cnn_output - self.combinatorial_output)
        return new_grads
    def propogate(self, input, labels):
        self.labels = labels
        self.forward_pass(input)
        grads = self.backward_pass()
        with torch.no_grad():
            for param, grad in zip(self.model.parameters(), grads):
                param -= grad
        return self.backward_pass()
    



def get_model(cfg, warcraft_experiment=True):
    #if we use warcraft experiment I will set the default parameters from the paper

    if warcraft_experiment:
        resnet_module = CombRenset18(out_features=144, in_channels=96)
        return resnet_module



def forward_pass(input, solver=dijkstra): # Include this fn in the architecture of the model
    input = input.detach().cpu().numpy()
    output = solver(input) # Inputs for dijkstra algo
    log_input_output(input, output) # decide how to save params!!
    
    return output # What is the correct form for CNN?

def backward_pass(grad, lambda_val, solver=dikkstra): # Include this fn in the architecture of the model
    input, output = load_input_output() # from forward pass after applying cc. Output: 
    input += param * grad
    perturbed_output = solver(input)

    gradient = -(1/lambda_val) * (perturbed_output - output)

    return gradient.to(device)

# class CNNModel(nn.Module):
#     def __init__(self, cfg):
#         super(CNNModel, self).__init__()
        
