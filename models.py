import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

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
        
        self.affr1 = nn.Linear(nhid['gc'][1], nhid['affr'][0])
        self.affr2 = nn.Linear(nhid['affr'][0], nhid['affr'][1])
        self.affr3 = nn.Linear(nhid['affr'][1], nfeat)
        self.dropout = dropout

    def forward(self, x, adj): #x:feature, adj:adjency matrix
        x1 = torch.tanh(self.gc1(x, adj)) 
        #x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = torch.tanh(self.gc2(x1, adj))
        #x2 = F.dropout(x2, self.dropout, training=self.training)
        Zn = x2.cuda().cpu().detach().numpy().copy()
       

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
        
        return [xc5, xr5], Zn

class GCN(nn.Module):
    def __init__(self, nfeat, nhid):
        super(GCN, self).__init__()

        #nhid = [512, 200]
        self.gc1 = GraphConvolution(nfeat, nhid[0])
        self.gc2 = GraphConvolution(nhid[0], nhid[1])

    def forward(self, x, adj): #x:特徴行列, adj:隣接行列として渡される
        x1 = torch.tanh(self.gc1(x, adj)) #第1層目のGCクラスのforward実行➡relu実行
        x2 = torch.tanh(self.gc2(x1, adj))
        
        return x2