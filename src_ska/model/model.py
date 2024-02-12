import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.utils.loss import HammingLoss
from math import sqrt
from .combinatorial_solvers import Dijkstra, DijskstraClass
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Put on every file
def glorot_initializer(shape):
    # Calculate the scale factor for Glorot initialization
    fan_in, fan_out = shape[0], shape[1]
    scale = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out), dtype=torch.float32))
    
    # Generate random values with a uniform distribution
    return torch.rand(shape, dtype=torch.float32) * 2 * scale - scale

def get_model(cfg, warcraft_experiment=True):
    #if we use warcraft experiment I will set the default parameters from the paper
    if warcraft_experiment:
        resnet_module = CombRenset18(out_features=144, in_channels=3,
                                     cfg=cfg)
        return resnet_module


class CombRenset18(nn.Module):

    def __init__(self, out_features, in_channels, cfg):
        """
        Expected shape is [b, 3, h, w]
        """
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(pretrained=False, num_classes=out_features)
        del self.resnet_model.conv1

        self.resnet_model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        output_shape = (int(sqrt(out_features)), int(sqrt(out_features)))
        self.pool = nn.AdaptiveMaxPool2d(output_shape)
        #self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)
        self.linear = nn.Linear(48, 64) # I HARDCODED THE NUMBERS # Fully connected layer to do dropout on

        #self.concrete_dropout = ConcreteDropout(self.linear) # Input the previous layer into Concrete Dropout to get its weights

        self.combinatorial_solver = DijskstraClass.apply
        self.grad_approx = GradientApproximator.apply

        #give us a normal convolution with the same shape as the input
        self.conv1_t = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.conv2_t = nn.Conv2d(12, 12, kernel_size=3, padding)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.cfg = cfg


    def forward(self, x):
        x = self.resnet_model.conv1(x) #64, 48, 48
        x = self.bn1(x)
        x = self.relu1(x) #64, 48, 48
        x = self.resnet_model.layer1(x)
        x = self.resnet_model.maxpool(x)

        x = self.pool(x) #64, 12, 12
        
        x = x.mean(dim=1)
        cnn_output = x.abs()

        if self.cfg.normalise:
            pass
            #we want here to unnormalise the cnn_output. Essentially shifting by mean and multiplying by std

        combinatorial_solver_output = self.combinatorial_solver(cnn_output)
        x = self.grad_approx(combinatorial_solver_output, cnn_output)

        return x, cnn_output #shape is 32, 12, 12

class GradientApproximator(torch.autograd.Function):
    def __init__(self, model, input_shape,
                 lambda_val=0.1):
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
    def forward(ctx, combinatorial_solver_output, cnn_output,
                ):
        #ctx.criterion = HammingLoss().requires_grad_(True)

        #ctx.labels = labels
        #ctx.cnn_input = cnn_input
        
        #this line here is the issue, there is no connection between the cnn_input and the combinatorial solver
        #combinatorial_output = ctx.combinatorial_solver(cnn_input)
        #loss = ctx.criterion(combinatorial_output, ctx.labels)
        #print("doing the forward pass")
        #import pdb; pdb.set_trace()
        ctx.save_for_backward(combinatorial_solver_output, cnn_output)
        return combinatorial_solver_output
    
    @staticmethod
    def backward(ctx, grad_input,
                 ): # Deviation from paper algo, calculate grad in function
        #shape of loss grad is [1, 32, 12, 12]
       # combinatorial_solver = DijskstraClass()
        #import pdb; pdb.set_trace()
        
        #return grad_input, grad_input
        lambda_val = 0.1
        combinatorial_solver_output, cnn_output = ctx.saved_tensors
        perturbed_cnn_weights = cnn_output + torch.multiply(10.0, grad_input)
        #torch.matmul(torch.full(cnn_output.shape, 10.0), grad_input) # Is this variable named accurately?
        perturbed_cnn_output = DijskstraClass.apply(perturbed_cnn_weights)
        new_grads = -(1 / 10) * (combinatorial_solver_output - perturbed_cnn_output)
        #import pdb; pdb.set_trace()
        return new_grads, new_grads
        # perturbed_cnn_output = combinatorial_solver(perturbed_cnn_weights)
        # new_grads = -(1 / ctx.lambda_val) * (combinatorial_solver_output - perturbed_cnn_output)
    
        # return new_grads
    



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
