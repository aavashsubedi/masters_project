import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_grad_flow(named_parameters, second_model_params = None):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for name, param in named_parameters:
        if(param.requires_grad) and ("bias" not in name) and (param.grad is not None):
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().cpu().numpy())
            max_grads.append(param.grad.abs().max().cpu().numpy())
    if second_model_params:
        for name, param in second_model_params:
            if(param.requires_grad) and ("bias" not in name) and (param.grad is not None):
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().cpu().numpy())
                max_grads.append(param.grad.abs().max().cpu().numpy())
    #go to layers and take the first part before the first "."
    layers = [layer.split(".")[0] for layer in layers]
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.2) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    #save the plot
    plt.savefig("/share/nas2/asubedi/masters_project/outputs/gradients/grad_flow.png")
    plt.show(block=True)