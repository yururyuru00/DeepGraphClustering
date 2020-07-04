#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        #hidden_layer=[128, 64, 32, 64, 32]
        self.gc1 = GraphConvolution(nfeat, nhid['gc'][0])
        self.gc2 = GraphConvolution(nhid['gc'][0], nhid['gc'][1])
        
        self.affc1 = nn.Linear(nhid['gc'][0] + nhid['gc'][1], nhid['affc'][0])
        self.bn1 = nn.BatchNorm1d(nhid['affc'][0])
        self.affc2 = nn.Linear(nhid['affc'][0], nhid['affc'][1])
        self.bn2 = nn.BatchNorm1d(nhid['affc'][1])
        self.affc3 = nn.Linear(nhid['affc'][1], nclass)
        
        self.affr1 = nn.Linear(nhid['gc'][0] + nhid['gc'][1], nhid['affr'][0])
        self.affr2 = nn.Linear(nhid['affr'][0], nhid['affr'][1])
        self.affr3 = nn.Linear(nhid['affr'][1], nfeat)
        self.dropout = dropout

    def forward(self, x, adj): #x:特徴行列, adj:隣接行列として渡される
        x1 = torch.tanh(self.gc1(x, adj)) #第1層目のGCクラスのforward実行➡relu実行
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = torch.tanh(self.gc2(x1, adj))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = torch.cat([x1, x2], dim=1)
        Zn = x2.cuda().cpu().detach().numpy().copy() #GCによるノード表現を保持
        #forward で得られる複数のx1~xkを連結したものを出力すればよいはず(まだ未実装)
        
        #Clustering MLP + BatchNormalization
        xc3 = F.relu(self.bn1(self.affc1(x2)))
        xc4 = F.relu(self.bn2(self.affc2(xc3)))
        xc5 = self.affc3(xc4)
        xc5 = F.log_softmax(xc5, dim=1)
        #Reconstruct MLP
        xr3 = F.relu(self.affr1(x2))
        xr3 = F.dropout(xr3, self.dropout, training=self.training)
        xr4 = F.relu(self.affr2(xr3))
        xr4 = F.dropout(xr4, self.dropout, training=self.training)
        xr5 = self.affr3(xr4)
        xr5 = F.dropout(xr5, self.dropout, training=self.training)
        #RMLPについては最後にdropoutは入れるべきかどうか考慮中
        
        return [xc5, xr5], Zn

