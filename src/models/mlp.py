"""
Multi Layered Perceptron for testing continual learning approaches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging as lg


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, act='relu', use_bn=False):
        super(LinearLayer, self).__init__()
        self.use_bn = use_bn
        self.lin = nn.Linear(input_dim, output_dim)
        self.act = nn.ReLU() if act == 'relu' else act
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
    def forward(self, x):
        if self.use_bn:
            return self.bn(self.act(self.lin(x)))
        out = self.lin(x)
        out = self.act(out)
        return out

class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.shape[0], -1)


class BaseModel(nn.Module):
    
    def __init__(self, num_inputs, num_hidden, num_outputs, normalize=True):
        super(BaseModel, self).__init__()
        self.normalize = normalize
        self.f1 = Flatten()
        self.lin1 = LinearLayer(num_inputs, num_hidden, use_bn=False)
        self.lin2 = LinearLayer(num_hidden, num_hidden, use_bn=False)
        self.lin3 = nn.Linear(num_hidden, num_outputs)
        
    def forward(self, x):
        out = self.f1(x)
        out = self.lin1(out)
        out = self.lin2(out)
        out = self.lin3(out)
        
        if self.normalize:
            out = F.normalize(out, dim=1)

        return out