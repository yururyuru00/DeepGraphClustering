import math
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid

from utilities import accuracy, nmi, purity
from models import GCN
from layers import FrobeniusNorm


loss_f = FrobeniusNorm()

def train(args, epoch, data, model, optimizer, log):
    model.train()
    node_emb = model(data.x, data.edge_index)
    loss = loss_f(node_emb, dane_emb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # logging
    log['loss'].append(loss.cuda().cpu().detach().numpy().copy())
    if(epoch % 10 == 0):
        Zn_np = node_emb.cuda().cpu().detach().numpy().copy()
        np.save('./data/experiment/test/Zn_epoch{}'.format(epoch), Zn_np)
        
        k_means = KMeans(args.n_class, n_init=10, random_state=0, tol=0.0000001)
        k_means.fit(Zn_np)

        label = data.y.cuda().cpu().detach().numpy().copy()
        nmi = normalized_mutual_info_score(label, k_means.labels_)
        log['nmi'].append(nmi)


# settingargs check
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='dataset of {Cora, Citeseer, Pubmed} (default: Cora)')
parser.add_argument('--n_class', type=int, default=7,
                    help='number of class (default: 7)')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--hidden', type=list, default=[512, 200],
                    help='Number of hidden units.')
parser.add_argument('--hidden', type=int, nargs='+', default=[512, 200],
                    help='number of hidden layer of GCN')
args = parser.parse_args()

# Load data
os.makedirs('./data/experiment/test')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='./data/experiment/', name=args.dataset)
data = dataset[0].to(device)
dane_emb = np.loadtxt('./data/experiment/DANEemb.csv')
dane_emb = torch.FloatTensor(dane_emb).to(device)

# Model and optimizer
n_attributes = data.x.shape[1]
model = GCN(n_attributes, args.hidden).to(device)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# Train and Save model
log = {'loss': [], 'nmi': []}
for epoch in tqdm(range(args.epochs+1)):
    train(args, epoch, data, model, optimizer, log)
torch.save(model.state_dict(), 'pretrained_gcn_reconstruct')


# plot log
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

with open('.data/experiment/test/log.txt', 'w') as w:
    for arg in vars(args):
        w.write('{}: {}\n'.format(arg, getattr(args, arg)))