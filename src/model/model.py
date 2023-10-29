import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from src.utils.loss import HammingLoss
from math import sqrt
from .combinatorial_solvers import Dijkstra, DijskstraClass
from src.utils.concrete_dropout import ConcreteDropout
import time

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
        resnet_module = CombRenset18(out_features=144, in_channels=3, cfg=cfg)
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

        self.concrete_dropout = ConcreteDropout(self.resnet_model.conv1) # Input the previous layer into Concrete Dropout to get its weights

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
        x = self.relu1(x) #64, 48, 48
        x = self.pool(x) #64, 12, 12
        x = self.concrete_dropout(x) # This does dropout on a convolutional layer !!!! Check if this is ok. Must use small amount of dropout

        x = x.mean(dim=1)
        cnn_output = x.abs()

        if self.cfg.normalise:
            pass
            #we want here to unnormalise the cnn_output. Essentially shifting by mean and multiplying by std

        combinatorial_solver_output = self.combinatorial_solver(cnn_output)
        x = self.grad_approx(combinatorial_solver_output, cnn_output)

        return x, cnn_output #shape is 32, 12, 12
    

class GradientApproximator(torch.autograd.Function):
    def __init__(self, model, input_shape, lambda_val=20):
        self.input = None
        self.output = None
        self.prev_cnn_input = torch.rand(input_shape)

        self.curr_output = None
        self.lambda_val = 20
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
       
        lambda_val = 20

        combinatorial_solver_output, cnn_output = ctx.saved_tensors
        perturbed_cnn_weights = cnn_output + torch.multiply(lambda_val, grad_input)
        #t0 = time.time()
        perturbed_cnn_output = DijskstraClass.apply(perturbed_cnn_weights) # 0.8s
        #t1 = time.time()
        #print(t1-t0)

        new_grads = -(1 / lambda_val) * (combinatorial_solver_output - perturbed_cnn_output)

        return new_grads, new_grads