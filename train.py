#!/usr/bin/env python
# coding: utf-8

# In[6]:


from __future__ import division
from __future__ import print_function

import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, kmeans
from models import GCN
from layers import FrobeniusNorm

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=120,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=np.array, default=[256, 128, 64, 32],
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args(args=[])
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
loss_fro = FrobeniusNorm()
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay) #lrが学習係数

if args.cuda: #cpuがかgpuどちらのtensorを使うかの処理
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

def train(epoch, loss1, loss2, acc):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    [output1, output2], Zn = model(features, adj)
    #features:特徴行列, adj:隣接行列を渡してforward実行
    #kmeans_labels = kmeans(Zn, torch.max(labels)+1)
    loss_train1 = F.nll_loss(output1[idx_train], labels[idx_train]) 
    #loss_train1はtensor型　nll_lossはsoftmax無しのクロスエントロピーのみのロス関数
    loss_train2 = loss_fro(output2[idx_train], features[idx_train]) #自作損失関数
    loss_train1.backward()
    #loss_train2.backward()
    optimizer.step()
    acc_train = accuracy(output1[idx_train], labels[idx_train])
    
    loss1.append(loss_train1)
    loss2.append(loss_train2)
    acc.append(acc_train)

    if not args.fastmode: #defaltはFalseなのでここの処理は行う
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        [output1, output2], Zn = model(features, adj)
    
    #kmeans_labels = kmeans(Zn, torch.max(labels)+1)
    loss_val1 = F.nll_loss(output1[idx_val], labels[idx_val])
    #loss_val2 = loss_fro(output2[idx_val], features[idx_val])
    acc_val = accuracy(output1[idx_val], labels[idx_val])
    print('Epoch:{:04d}'.format(epoch+1),
          'lss1_train: {:.4f}'.format(loss_train1.item()),
          'lss2_train: {:.4f}'.format(loss_train2.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'lss_val: {:.4f}'.format(loss_val1.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    [output1, output2], Zn = model(features, adj)
    np.savetxt('D:\python\GCN\DeepGraphClustering\data\experiment\Zn_labelbased.csv', Zn)
    #kmeans_labels = kmeans(Zn, torch.max(labels)+1)
    loss_test1 = F.nll_loss(output1[idx_test], labels[idx_test])
    loss_test2 = loss_fro(output2[idx_test], features[idx_test]) #自作損失関数
    acc_test1 = accuracy(output1[idx_test], labels[idx_test])
    print("Test set results:",
          "lss1= {:.4f}".format(loss_test1.item()),
          "lss2= {:.4f}".format(loss_test2.item()),
          "accuracy= {:.4f}".format(acc_test1.item()))


# Train model
t_total = time.time()
loss1, loss2, acc = [], [], []
for epoch in range(args.epochs):
    train(epoch, loss1, loss2, acc)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

fig = plt.figure(figsize=(32, 16))
ax1, ax2, ax3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)
ax1.plot(loss1)
ax2.plot(loss2)
ax3.plot(acc)
#plt.savefig('D:\python\GCN\DeepGraphClustering\data\experiment\loss+acc_clusterbased.png')

# Testing
test()


# In[ ]:




