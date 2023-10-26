"""
Implemenation of 'Concrete Dropout' (Gal et al.) https://doi.org/10.48550/arXiv.1705.07832.
This is the continuous analog to the usual discrete form of dropout.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcreteDropout(nn.Module) : # This is a Pytorch custom dropout layer
    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5) :
        super(ConcreteDropout, self).__init__(layer)

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.p_logit = nn.Parameter(torch.Tensor(1))
        self.p_logit.data.uniform_(-2., 0.)
        self.p = torch.sigmoid(self.p_logit)


    def forward(self, x):
        eps = 1e-7        # Small factor epsilon
        temp = 1.0 / 10.0 # 'temperature' of distribution
        unif_noise = torch.rand_like(x)
        drop_prob = torch.sigmoid((torch.log(self.p + eps) - torch.log(1 - self.p + eps) +
                     torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps)) / temp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p
        x = x * random_tensor  # Apply dropout
        x = x / retain_prob  # Adjust for dropout rate

        # Calculate the regularizers
        input_dim = x.size(1)
        weight_regularizer = self.weight_regularizer * torch.sum(self.layer.weight**2) / (1. - self.p)

        dropout_regularizer = (self.p * torch.log(self.p) + ((1. - self.p) * torch.log(1. - self.p)))
        dropout_regularizer *= (self.dropout_regularizer * input_dim)

        regularizer = weight_regularizer + dropout_regularizer

        return x, regularizer
