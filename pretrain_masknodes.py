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
from sklearn.metrics.cluster import normalized_mutual_info_score
import itertools

from utilities import Mask, GraphAugmenter, ExtractAttribute, nmi
from models import GCN
from layers import NeuralTensorNetwork
from debug import plot_Zn

criterion1 = torch.nn.BCEWithLogitsLoss()
criterion2 = torch.nn.BCEWithLogitsLoss()

def train(args, epoch, data, models, optimizers, log):
    model, linear_pred_nodes, linear_pred_edges = models
    optimizer, optimizer_linear_nodes, optimizer_linear_edges = optimizers

    model.train()
    linear_pred_nodes.train()
    linear_pred_edges.train()

    node_rep = model(data.x, data.edge_index)

    loss = torch.tensor(0.).float().to(device=data.x.device.type)
    # mask the node representation
    if(args.mask_rate_node > 0.):
        pred_nodes = linear_pred_nodes(node_rep[data.masked_node_idxes])
        loss += criterion1(pred_nodes.double(), data.masked_node_label)

    # mask the edge representation
    if(args.mask_rate_edge > 0.):
        num_nodes = data.x.size()[0]
        edge_preds = []
        for u, v in data.mask_edge_idxes:
            edge_pred = linear_pred_edges(node_rep[u], node_rep[v])
            edge_preds.append(edge_pred)
        edge_preds = torch.cat(edge_preds, axis=0)
        loss += criterion2(edge_preds, data.mask_edge_label)
    
    optimizer.zero_grad()
    optimizer_linear_nodes.zero_grad()
    optimizer_linear_edges.zero_grad()

    loss.backward()

    optimizer.step()
    optimizer_linear_nodes.step()
    optimizer_linear_edges.step()

    # log and debug
    log['loss'].append(loss)

    if(epoch % 10 == 0):
        Zn_np = node_rep.cuda().cpu().detach().numpy().copy()
        label = data.y.cuda().cpu().detach().numpy().copy()
        plot_Zn(
            Zn_np, label, path_save='./data/experiment/test/Zn_masknode_epoch{}'.format(epoch))
        
        n_class = torch.max(data.y).cuda().cpu().detach().numpy().copy() + 1
        k_means = KMeans(n_class, n_init=10, random_state=0, tol=0.0000001)
        k_means.fit(Zn_np)
        nmi = normalized_mutual_info_score(
              data.y.cuda().cpu().detach().numpy().copy(), k_means.labels_)
        log['nmi'].append(nmi)

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
parser.add_argument('--mask_rate_edge', type=float, default=0.15,
                    help='mask edges ratio (default: 0.15)')
parser.add_argument('--hidden', type=list, default=[128, 64, 32, 16],
                    help='number of hidden layer of GCN for substract representation')
args = parser.parse_args()

# load and transform dataset
os.makedirs('./data/experiment/test_')
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
linear_pred_nodes = torch.nn.Linear(dim_emb, n_attributes).to(device)
linear_pred_edges = NeuralTensorNetwork(dim_emb, 1).to(device)

# set up optimizer for the GNNs
optimizer = optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.decay)
optimizer_linear_nodes = optim.Adam(
    linear_pred_nodes.parameters(), lr=args.lr, weight_decay=args.decay)
optimizer_linear_edges = optim.Adam(
    linear_pred_edges.parameters(), lr=args.lr, weight_decay=args.decay)

models = [model, linear_pred_nodes, linear_pred_edges]
optimizers = [optimizer, optimizer_linear_nodes, optimizer_linear_edges]

# train
log = {'loss': [], 'nmi': []}
for epoch in tqdm(range(args.epochs)):
    loss = train(args, epoch, data, models, optimizers, log)
torch.save(model.state_dict(), 'pretrained_gcn')

fig = plt.figure(figsize=(35, 35))
ax1, ax2 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)
ax1.plot(log['loss'], label='loss')
ax1.legend(loc='upper right', prop={'size': 25})
ax1.tick_params(axis='x', labelsize='23')
ax1.tick_params(axis='y', labelsize='23')
ax2.plot(log['nmi'], label='nmi')
ax2.legend(loc='upper left', prop={'size': 25})
ax2.tick_params(axis='x', labelsize='23')
ax2.tick_params(axis='y', labelsize='23')
plt.savefig('./data/experiment/test/result.png')

