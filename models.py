import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DGC(nn.Module):
    def __init__(self, base, n_feat, n_hid, n_class, dropout):
        super(DGC, self).__init__()

        # add GClayer based on pretrained GCN
        self.gc_layers = torch.nn.ModuleList()
        for layer in base.gc_layers:
            self.gc_layers.append(layer)

        # add clustering layer
        self.clus1 = nn.Linear(n_hid['gc'][-1], n_hid['clustering'][0])
        self.bn1 = nn.BatchNorm1d(n_hid['clustering'][0])
        self.clus2 = nn.Linear(n_hid['clustering'][0], n_hid['clustering'][1])
        self.bn2 = nn.BatchNorm1d(n_hid['clustering'][1])
        self.clus3 = nn.Linear(n_hid['clustering'][1], n_class)
        self.softmax = nn.Softmax(dim=1)

        # add reconstruct layer
        self.rec1 = nn.Linear(n_hid['gc'][-1], n_hid['reconstruct'][0])
        self.rec2 = nn.Linear(n_hid['reconstruct'][0], n_hid['reconstruct'][1])
        self.rec3 = nn.Linear(n_hid['reconstruct'][1], n_feat)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Graph Convolution
        n_layer_gc = len(self.gc_layers)
        for layer in range(n_layer_gc):
            x = self.gc_layers[layer](x, edge_index)
            x = torch.tanh(x)
        Zn = x.cuda().cpu().detach().numpy().copy()

        # Clustering MLP
        x_c = F.relu(self.bn1(self.clus1(x)))
        x_c = F.relu(self.bn2(self.clus2(x_c)))
        x_c = self.clus3(x_c)
        x_c = self.softmax(x_c)

        # Reconstruct MLP
        x_r = F.relu(self.rec1(x))
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_r = F.relu(self.rec2(x_r))
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_r = self.rec3(x_r)
        x_r = F.dropout(x_r, self.dropout, training=self.training)

        return [x_c, x_r], Zn


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
