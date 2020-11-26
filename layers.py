import math
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class NeuralTensorNetwork(Module):
    def __init__(self, in_features, out_features):
        super(NeuralTensorNetwork, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.FloatTensor(in_features, in_features))
        self.weight2 = Parameter(torch.FloatTensor(2*in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight1)
        nn.init.uniform_(self.weight2, 0, 1)
        nn.init.uniform_(self.bias, 0, 1)

    def forward(self, input1, input2):
        input1_ = torch.matmul(input1,self.weight1)
        output1 = torch.matmul(input1_, input2)
        output2 = torch.dot(self.weight2, torch.cat([input1, input2], axis=0))
        
        return torch.tanh(output1 + output2 + self.bias)


class FrobeniusNorm(Module):
    def __init__(self):
        super(FrobeniusNorm, self).__init__()

    def forward(self, inputs, targets):
        diff = inputs - targets
        return torch.norm(diff)


class HardClusterLoss(Module):
    def __init__(self):
        super(HardClusterLoss, self).__init__()

    def forward(self, inputs):
        # return rate*torch.norm(inputs) #引数にrateを実装すること
        return torch.norm(inputs)