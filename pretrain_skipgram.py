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

from utilities import ExtractSubstructureContextPair, nmi
from models import GCN
from debug import plot_Zn


def train(epoch, model_substruct, model_context, data, optimizer_substruct, optimizer_context, log):
    model_substruct.train()
    model_context.train()

    n_class = len(data.list)
    class_labels = np.arange(n_class)
    map_postitive_negative_idx = np.roll(class_labels, 1)
    for c_i in range(n_class):

        # creating substract, context, and negative representations
        representations = model_substruct(
            data.x, data.edge_index)
        substruct_rep = representations[data.list[c_i].center_idx]
        negative_rep = representations[data.list[map_postitive_negative_idx[c_i]].center_idx].reshape(
            1, -1)
        context_rep = model_context(data.list[c_i].x_context, data.list[c_i].edge_index_context)[
            data.list[c_i].context_idxes]

        if(epoch % 10 == 0 and c_i == 0):  # debug
            Zn_np = representations.cuda().cpu().detach().numpy().copy()
            label = data.y.cuda().cpu().detach().numpy().copy()
            plot_Zn(
                Zn_np, label, path_save='./data/experiment/test/Zn_skipgram_epoch{}'.format(epoch))

            n_class = torch.max(data.y).cuda(
            ).cpu().detach().numpy().copy() + 1
            k_means = KMeans(n_class, n_init=10, random_state=0, tol=0.0000001)
            k_means.fit(Zn_np)
            nmi = clus.adjusted_mutual_info_score(
                k_means.labels_, data.y.cuda().cpu().detach().numpy().copy(), "arithmetic")
            log['nmi'].append(nmi)

        # skig gram with negative sampling
        pred_pos = torch.sum(substruct_rep*context_rep, dim=1)
        pred_neg = torch.sum(substruct_rep*negative_rep, dim=1)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss_pos = criterion(pred_pos.double(), torch.ones(
            len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(
            len(pred_neg)).to(pred_neg.device).double())

        optimizer_substruct.zero_grad()
        optimizer_context.zero_grad()
        loss = loss_pos + loss_neg
        loss.backward()
        optimizer_substruct.step()
        optimizer_context.step()

    return float(loss.detach().cpu().item())


parser = argparse.ArgumentParser(
    description='PyTorch implementation of pre-training of GNN')
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--n_class', type=int, default=2,
                    help='number of class')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (defalt: 100)')
parser.add_argument('--border', type=int, default=1,
                    help='boderline between substract and context graph (default: 3).')
parser.add_argument('--hidden1', type=list, default=[1024, 512, 256],
                    help='number of hidden layer of GCN for substract representation')
parser.add_argument('--hidden2', type=list, default=[512, 256],
                    help='number of hidden layer of GCN for context representation')
args = parser.parse_args()

# load and transform dataset
os.makedirs('./data/experiment/test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(args.dataset == 'KarateClub'):
    dataset = KarateClub(transform=ExtractSubstructureContextPair(
        args.n_class, args.border, device))
else:
    dataset = Planetoid(root='./data/experiment/', name=args.dataset,
                        transform=ExtractSubstructureContextPair(args.n_class, args.border, device))
data = dataset[0].to(device)

# set up GCN model
n_attributes = data.x.shape[1]
model_substruct = GCN(n_attributes, args.hidden1).to(device)
model_context = GCN(n_attributes, args.hidden2).to(device)

# set up optimizer for the two GNNs
optimizer_substruct = optim.Adam(
    model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
optimizer_context = optim.Adam(
    model_context.parameters(), lr=args.lr, weight_decay=args.decay)

# train
log = {'loss': [], 'nmi': []}
for epoch in tqdm(range(args.epochs)):
    loss = train(epoch, model_substruct, model_context, data,
                 optimizer_substruct, optimizer_context, log)
torch.save(model_substruct.state_dict(), 'pretrained_gcn')

# log
fig = plt.figure(figsize=(17, 17))
plt.plot(log['nmi'], label='nmi')
plt.legend(loc='upper right', prop={'size': 12})
plt.tick_params(axis='x', labelsize='12')
plt.tick_params(axis='y', labelsize='12')
plt.savefig('./data/experiment/test/result.png')
