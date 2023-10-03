import torch

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