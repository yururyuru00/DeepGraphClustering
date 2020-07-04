import math
import numpy as np
from numpy import linalg

import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        ##パラメータはnn.parameterオブジェクトとしないと，後で重みの更新ができない
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

    def forward(self, input, adj): #input:特徴行列Z, adj:隣接行列Aとして渡される
        support = torch.mm(input, self.weight) #畳み込み DAZWの内，ZW(=support)の演算
        output = torch.spmm(adj, support) #畳み込み DAZWの内，(DA)supportの演算
        if self.bias is not None: #今回はbiasあり
            return output + self.bias
        else:
            return output

class FrobeniusNorm(Module):
    def __init__(self):
        super(FrobeniusNorm, self).__init__()

    def forward(self, inputs, targets):
        diff = inputs - targets
        return torch.norm(diff)
    
class purity_loss(Module):
    def __init__(self):
        super(purity_loss, self).__init__()

    def forward(self, inputs, targets):
        size = torch.LongTensor([len(targets)])[0]
        inputs = torch.max(inputs, 1).indices.cuda().cpu().detach().numpy().copy()
        targets = targets.cuda().cpu().detach().numpy().copy()
        clus_size = int(np.max(inputs))+1
        clas_size = int(np.max(targets))+1
        table = torch.zeros(clus_size, clas_size)
        for i in range(size):
            table[inputs[i]][targets[i]] += 1
        sum = torch.zeros(1, requires_grad=True)
        for k in range(clus_size):
            sum += torch.max(table[k])
        return sum/size