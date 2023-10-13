import torch

device = torch.device("cpu" if torch.cuda.is_available() else "cpu") # Put on every file

import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from src.utils.loss import HammingLoss
from math import sqrt
from .combinatorial_solvers import Dijkstra

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
        return x #shape is 32, 12, 12
    

class GradientApproximator(torch.autograd.Function):
    def __init__(self, model, input_shape):
        self.input = None
        self.output = None
        self.prev_cnn_input = torch.rand(input_shape)

        self.curr_output = None
        self.lambda_val = 0.1
        self.model = model
        self.criterion = HammingLoss().requires_grad_(True)
        self.combinatorial_solver = Dijkstra()
        self.labels = None
        self.cnn_loss = None

    @staticmethod
    def forward(ctx, cnn_input, labels):
        #ctx.criterion = HammingLoss().requires_grad_(True)

        ctx.labels = labels
        ctx.cnn_input = cnn_input
        
        #this line here is the issue, there is no connection between the cnn_input and the combinatorial solver
        ctx.combinatorial_output = ctx.combinatorial_solver(cnn_input)
        ctx.loss = ctx.criterion(ctx.combinatorial_output, ctx.labels)
        print("doing the forward pass")
        import pdb; pdb.set_trace()
        return ctx.combinatorial_output
    
    @staticmethod
    def backward(ctx): # Deviation from paper algo, calculate grad in function
        ctx.loss_grad = torch.autograd.grad(ctx.loss, ctx.combinatorial_output)
       # import pdb ; pdb.set_trace()
        perturbed_cnn_weights = ctx.cnn_input + torch.matmul(torch.full(ctx.cnn_input.shape, 0.1), ctx.loss_grad[0]) # Is this variable named accurately?
        perturbed_cnn_output = ctx.combinatorial_solver(perturbed_cnn_weights)
        new_grads = -(1/ctx.lambda_val) * (perturbed_cnn_output - ctx.combinatorial_solver(ctx.cnn_input))
       # import pdb; pdb.set_trace()
        return new_grads
    



# class GradientApproximator(torch.autograd.Function):

#     def __init__(self, model, input_shape):
#         self.input = None
#         self.output = None
#         self.prev_cnn_input = torch.rand(input_shape)

#         self.curr_output = None
#         self.lambda_val = 0.1
#         self.model = model
#         self.loss = HammingLoss()
#         self.loss.requires_grad_(True)
#         self.combinatorial_solver = Dijkstra
#         self.labels = None
#         self.cnn_loss = None
#     @staticmethod
#     def forward(self, cnn_input, labels, solver=Dijkstra()):
#         self.labels = labels
#             #take the shape of the input and create a tensor of random numbers with the same shape

#         self.cnn_input = cnn_input
#         self.combinatorial_output = solver(self.cnn_input)
#        # self.prev_cnn_input = self.output
#         return self.combinatorial_output
#     @staticmethod
#     def backward(self):
#         # self.combinatorial_output.require_grad = True
#         # self.labels.require_grad = False
#         #input to the backward pass is from the forward pass before djikstra and after djikstra
#         loss = self.loss(self.combinatorial_output, self.labels)
#         loss.required_grad = True
#         self.model.eval()
#         loss.backward()
#         loss_grad = self.combinatorial_output.grad
#         # import pdb; pdb.set_trace()
#         perturbed_cnn_weights   = self.prev_cnn_input + self.lambda_val * loss_grad
#         perturbed_cnn_output    = Dijkstra(perturbed_cnn_weights)
#         new_grads = -(1/self.lambda_val) * (perturbed_cnn_output - self.combinatorial_output)
#         return new_grads
    # def propogate(self, cnn_input, labels):
    #     self.labels = labels
    #     self.forward_pass(cnn_input)
    #     grads = self.backward_pass()
    #     import pdb; pdb.set_trace()

        # for name, param in self.model.named_parameters():
        #     import pdb; pdb.set_trace()

# def forward_pass(input, solver=dijkstra): # Include this fn in the architecture of the model
#     input = input.detach().cpu().numpy()
#     output = solver(input) # Inputs for dijkstra algo
#     log_input_output(input, output) # decide how to save params!!
    
#     return output # What is the correct form for CNN?

# def backward_pass(grad, lambda_val, solver=dikkstra): # Include this fn in the architecture of the model
#     input, output = load_input_output() # from forward pass after applying cc. Output: 
#     input += param * grad
#     perturbed_output = solver(input)

#     gradient = -(1/lambda_val) * (perturbed_output - output)

#     return gradient.to(device)

# class CNNModel(nn.Module):
#     def __init__(self, cfg):
#         super(CNNModel, self).__init__()
