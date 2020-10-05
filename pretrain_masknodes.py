import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
import torch.optim as optim
import os
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as clus
import itertools

from utilities import Mask, nmi
from models import GCN
from debug import plot_Zn

criterion1 = torch.nn.CrossEntropyLoss()
criterion2 = torch.nn.BCELoss()


def train(args, epoch, model, linear_pred_nodes, data,
          optimizer, optimizer_linear, log):
    model.train()
    linear_pred_nodes.train()

    node_rep = model(data.x, data.edge_index)
    num_nodes = data.x.size()[0]

    if(epoch % 10 == 0):
        Zn_np = node_rep.cuda().cpu().detach().numpy().copy()
        label = data.y.cuda().cpu().detach().numpy().copy()
        plot_Zn(
            Zn_np, label, path_save='./data/experiment/test/Zn_masknode_epoch{}'.format(epoch))

    # mask the node representation
    if(args.mask_rate_node > 0.):
        pred_nodes = linear_pred_nodes(node_rep[data.masked_node_indices])
        loss = criterion1(pred_nodes.double(), data.mask_node_label)

    # mask the edge representation
    if(args.mask_rate_edge > 0.):
        for i, (u, v) in enumerate(itertools.combinations(range(num_nodes), 2)):
            edge_rep = node_rep[u] + node_rep[v]
            edge_pre = linear_pred_nodes(edge_rep)

            loss += criterion2(edge_pre.double(), data.edge_label[i].view(-1))

    optimizer.zero_grad()
    optimizer_linear.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer_linear.step()

    # log and debug
    log['loss'].append(loss)

    return float(loss.detach().cpu().item())


parser = argparse.ArgumentParser(
    description='PyTorch implementation of pre-training of GNN')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--n_class', type=int, default=7,
                    help='number of class')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (defalt: 100)')
parser.add_argument('--mask_rate_node', type=float, default=0.15,
                    help='mask nodes ratio (default: 0.15)')
parser.add_argument('--mask_rate_edge', type=float, default=0.00,
                    help='mask nodes ratio (default: 0.00)')
parser.add_argument('--hidden', type=list, default=[7, 7, 7],
                    help='number of hidden layer of GCN for substract representation')
args = parser.parse_args()

# load and transform dataset
os.makedirs('./data/experiment/test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(args.dataset == 'KarateClub'):
    dataset = KarateClub(transform=Mask(
        args.mask_rate_node, args.mask_rate_edge))
else:
    dataset = Planetoid(root='./data/experiment/', name=args.dataset,
                        transform=Mask(args.mask_rate_node, args.mask_rate_edge))
data = dataset[0].to(device)

# set up GCN model and linear model to predict node features
n_attributes = data.x.shape[1]
model = GCN(n_attributes, args.hidden).to(device)

dim_emb = args.hidden[-1]
# below linear model predict if edge between nodes is exist or not
linear_pred_nodes = torch.nn.Linear(dim_emb, args.n_class).to(device)

# set up optimizer for the GNNs
optimizer = optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.decay)
optimizer_linear = optim.Adam(
    linear_pred_nodes.parameters(), lr=args.lr, weight_decay=args.decay)

# train
log = {'loss': []}
for epoch in tqdm(range(args.epochs)):
    loss = train(args, epoch, model, linear_pred_nodes, data,
                 optimizer, optimizer_linear, log)
torch.save(model.state_dict(), 'pretrained_gcn')

# log
fig = plt.figure(figsize=(17, 17))
plt.plot(log['loss'], label='loss')
plt.legend(loc='upper right', prop={'size': 12})
plt.tick_params(axis='x', labelsize='12')
plt.tick_params(axis='y', labelsize='12')
plt.savefig('./data/experiment/test/result.png')
