import torch

def forward_pass(input, solver=dijkstra): # Include this fn in the architecture of the model
    output = solver(input) # Inputs for dijkstra algo
    log_input_output(input, output) # decide how to save params!!
    
    return output

def backward_pass(grad, param, solver=dikkstra): # Include this fn in the architecture of the model
    input, output = load_input_output()
    input += param * grad
    perturbed_output = solver(input)

    return -(1/param) * (perturbed_output - output)