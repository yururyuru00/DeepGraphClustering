import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch_geometric.nn import GCNConv


class DGC(nn.Module):
    def __init__(self, base, nfeat, nhid, nclass, dropout):
        super(DGC, self).__init__()

        self.gc1 = base.gc1
        self.gc2 = base.gc2

        self.affc1 = nn.Linear(nhid['gc'][1], nhid['affc'][0])
        self.bn1 = nn.BatchNorm1d(nhid['affc'][0])
        self.affc2 = nn.Linear(nhid['affc'][0], nhid['affc'][1])
        self.bn2 = nn.BatchNorm1d(nhid['affc'][1])
        self.affc3 = nn.Linear(nhid['affc'][1], nclass)
        self.softmax = nn.Softmax(dim=1)

        self.affr1 = nn.Linear(nhid['gc'][1], nhid['affr'][0])
        self.affr2 = nn.Linear(nhid['affr'][0], nhid['affr'][1])
        self.affr3 = nn.Linear(nhid['affr'][1], nfeat)
        self.dropout = dropout

    def forward(self, x, adj):  # x:feature, adj:adjency matrix
        x1 = torch.tanh(self.gc1(x, adj))
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = torch.tanh(self.gc2(x1, adj))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        Zn = x2.cuda().cpu().detach().numpy().copy()

        # Clustering MLP + BatchNormalization
        xc3 = F.relu(self.bn1(self.affc1(x2)))
        xc4 = F.relu(self.bn2(self.affc2(xc3)))
        xc5 = self.affc3(xc4)
        xc5 = self.softmax(xc5)
        # Reconstruct MLP
        xr3 = F.relu(self.affr1(x2))
        xr3 = F.dropout(xr3, self.dropout, training=self.training)
        xr4 = F.relu(self.affr2(xr3))
        xr4 = F.dropout(xr4, self.dropout, training=self.training)
        xr5 = self.affr3(xr4)
        xr5 = F.dropout(xr5, self.dropout, training=self.training)

        return [xc5, xr5], Zn


class GCN(nn.Module):
    def __init__(self, n_layer, n_feat, hid):
        super(GCN, self).__init__()
        self.num_layer = n_layer

        self.gc_layers = torch.nn.ModuleList()
        for layer in range(n_layer):
            if(layer == 0):
                self.gc_layers.append(GCNConv(n_feat, hid[0]))
            else:
                self.gc_layers.append(GCNConv(hid[layer-1], hid[layer]))

    def forward(self, x, edge_index):
        for layer in range(self.num_layer):
            x = self.gc_layers[layer](x, edge_index)
        return x
