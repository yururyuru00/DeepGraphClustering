from __future__ import division
from __future__ import print_function

import math
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from utilities import load_data, accuracy, nmi, purity, kmeans
from models import GCN
from layers import FrobeniusNorm, purity_loss

def train(epoch, log):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = loss_f(output, dane_emb)
    loss.backward()
    optimizer.step()

    log['loss'].append(loss.cuda().cpu().detach().numpy().copy())

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

# settingargs check
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=list, default=[512, 200],
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, _, idx_train = load_data()  # pretrainではラベルは使わない
dane_emb = np.loadtxt('./data/experiment/DANEemb.csv')

# Model and optimizer
model = GCN(nfeat=features.shape[1], nhid=args.hidden)
loss_f = FrobeniusNorm()
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)  # lrが学習係数

if args.cuda:  # cpuかgpuどちらのtensorを使うかの処理
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    dane_emb = torch.FloatTensor(dane_emb).cuda()
    idx_train = idx_train.cuda()


# untrained Zn log
model.train()
output = model(features, adj)
np.savetxt('./data/experiment/Zn_untrainedGCN.csv',
           output.cuda().cpu().detach().numpy().copy())

# Train and Save model
t_total = time.time()
log = {'loss': [], 'nmi': []}
for epoch in range(args.epochs+1):
    train(epoch, log)
print("Optimization Finished!")
torch.save(model.state_dict(), 'model_gcn')

# trained Zn log
model.train()
output = model(features, adj)
np.savetxt('./data/experiment/Zn_trainedGCN.csv',
           output.cuda().cpu().detach().numpy().copy())

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
