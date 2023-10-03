import torch
<<<<<<< HEAD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file

def forward_pass(input, solver=dijkstra): # Include this fn in the architecture of the model
    input = input.detach().cpu().numpy()
    output = solver(input) # Inputs for dijkstra algo
    log_input_output(input, output) # decide how to save params!!
    
    return output

def backward_pass(grad, lambda_val, solver=dikkstra): # Include this fn in the architecture of the model
    input, output = load_input_output()
    input += param * grad
    perturbed_output = solver(input)

    gradient = -(1/lambda_val) * (perturbed_output - output)

    return gradient.to(device)
=======
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
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


def get_model(cfg, warcraft_experiment=True):
    #if we use warcraft experiment I will set the default parameters from the paper

    if warcraft_experiment:
        resnet_module = CombRenset18(out_features=144, in_channels=96)
        return resnet_module



# class CNNModel(nn.Module):
#     def __init__(self, cfg):
#         super(CNNModel, self).__init__()
        
>>>>>>> c537756ef4e0adfcb1e61a61f18ef892eb9e27ab
