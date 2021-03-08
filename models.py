import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv


class DGC(nn.Module):
    def __init__(self, base, n_feat, n_hid, n_class, dropout):
        super(DGC, self).__init__()
        self.gcn_layer = n_hid['gcn']
        self.clus_layer = n_hid['clustering']
        self.reconst_layer = n_hid['reconstruct']

        # add GCN layer based on pretrained GCN
        self.gc_layers = torch.nn.ModuleList()
        for layer in base.gc_layers:
            self.gc_layers.append(layer)

        # add clustering layer
        self.clus = nn.ModuleList()
        n_layers = [self.gcn_layer[-1]] + self.clus_layer
        for idx in range(len(n_layers)-1):
            self.clus.append(nn.Linear(n_layers[idx], n_layers[idx+1]))
            self.clus.append(nn.BatchNorm1d(n_layers[idx+1]))
            self.clus.append(nn.ReLU())
        self.clus.append(nn.Linear(n_layers[-1], n_class))
        self.clus.append(nn.Softmax(dim=1))
        
        # add reconstruct layer
        self.reconst = nn.ModuleList()
        n_layers = [self.gcn_layer[-1]] + self.reconst_layer
        for idx in range(len(n_layers)-1):
            self.reconst.append(nn.Linear(n_layers[idx], n_layers[idx+1]))
            self.reconst.append(nn.ReLU())
            self.reconst.append(nn.Dropout(p=dropout))
        self.reconst.append(nn.Linear(n_layers[-1], n_feat))
        self.reconst.append(nn.Dropout(p=dropout))

    def forward(self, x, edge_index):
        # convolution graph
        n_layer_gc = len(self.gc_layers)
        for layer in range(n_layer_gc):
            x = self.gc_layers[layer](x, edge_index)
            x = torch.tanh(x)

        x_c, x_r = x.clone(), x.clone()
        # cluster and reconstruct by MLP
        for layer in self.clus:
            x_c = layer(x_c)
        for layer in self.reconst:
            x_r = layer(x_r)

        return [x_c, x_r], x


class GCN(nn.Module):
    def __init__(self, model, n_feat, hid, dropout):
        torch.manual_seed(0)
        super(GCN, self).__init__()
        self.layers = [n_feat] + hid
        self.n_conv = len(self.layers)-1
        self.dropout = dropout
        
        self.gc_layers = torch.nn.ModuleList()
        for idx in range(self.n_conv):
            if(model == 'gcn'): # if use GCN Conv
                self.gc_layers.append(GCNConv(self.layers[idx], self.layers[idx+1]))

            else: # if use GIN Conv
                mlp = nn.Sequential(nn.Linear(self.layers[idx], self.layers[idx]), 
                                    torch.nn.BatchNorm1d(self.layers[idx]), nn.ReLU(),
                                    nn.Linear(self.layers[idx], self.layers[idx+1]))
                self.gc_layers.append(GINConv(mlp, eps=0., train_eps=False))

    def forward(self, x, edge_index):
        for idx in range(self.n_conv - 1):
            x = self.gc_layers[idx](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)

        # iff the last convolutional layer, we don't use relu and dropout
        x = self.gc_layers[-1](x, edge_index)

        return x