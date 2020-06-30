#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from utils import load_data, accuracy, nmi, kmeans
from models import GCN
from layers import FrobeniusNorm

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000,
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
labels_sclump = np.loadtxt('D:/python/GCN/DeepGraphClustering/data/experiment/sclump_label.csv')
labels_sclump = torch.LongTensor(labels_sclump).clone().to('cuda')

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

def train(epoch, log):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    [output1, output2], Zn = model(features, adj)
    #kmeans_labels = kmeans(Zn, torch.max(labels)+1)
    loss_train1 = F.nll_loss(output1[idx_train], labels_sclump[idx_train]) #クロスエントロピー
    loss_train2 = loss_fro(output2[idx_train], features[idx_train]) #自作損失関数
    loss_train1.backward() #更にbackwardならretain_graph = trueにすること
    #loss_train2.backward()
    optimizer.step()
    acc_train = nmi(output1[idx_train], labels[idx_train])
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    
    if not args.fastmode: #defaltはFalseなのでここの処理は行う
        # deactivates dropout during validation run.
        model.eval()
        [output1, output2], Zn = model(features, adj)  
    #kmeans_labels = kmeans(Zn, torch.max(labels)+1)
    loss_val1 = F.nll_loss(output1[idx_val], labels_sclump[idx_val])
    loss_val2 = loss_fro(output2[idx_val], features[idx_val])
    acc_val = nmi(output1[idx_val], labels[idx_val]) #acc_valはnmiの場合，npで返ってくる
    
    log['loss_train1'].append(loss_train1.cuda().cpu().detach().numpy().copy())
    log['loss_train2'].append(loss_train2.cuda().cpu().detach().numpy().copy()/len(idx_train))
    log['acc_train'].append(acc_train)
    log['loss_val1'].append(loss_val1.cuda().cpu().detach().numpy().copy())
    log['loss_val2'].append(loss_val2.cuda().cpu().detach().numpy().copy()/len(idx_val))
    log['acc_val'].append(acc_val)


def test(log):
    model.eval()
    [output1, output2], Zn = model(features, adj)
    np.savetxt('D:\python\GCN\DeepGraphClustering\data\experiment\Zn_sclump_1on2off.csv', Zn)
    kmeans_labels = kmeans(Zn, torch.max(labels)+1)
    acc_test = nmi(output1[idx_test], labels[idx_test])
    log['acc_test'] = acc_test


# Train model
t_total = time.time()
log = {'loss_train1':[], 'loss_train2':[], 'acc_train':[], 'loss_val1':[], 
       'loss_val2':[], 'acc_val':[], 'acc_test':0}
for epoch in range(args.epochs):
    train(epoch, log)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s\n".format(time.time() - t_total))

# Testing
test(log)

#log + plot
for epoch in range(args.epochs)[::10]:
    print('Epoch:{:04d}'.format(epoch),
          'lss1_train: {:.4f}'.format(log['loss_train1'][epoch]),
          'lss2_train: {:.4f}'.format(log['loss_train2'][epoch]),
          'acc_train: {:.4f}'.format(log['acc_train'][epoch]),
          'lss1_val: {:.4f}'.format(log['loss_val1'][epoch]),
          'lss2_val: {:.4f}'.format(log['loss_val2'][epoch]),
          'acc_val: {:.4f}'.format(log['acc_val'][epoch]))
print("#################\nTest set results:", "accuracy= {:.4f}".format(log['acc_test']))
fig = plt.figure(figsize=(32, 16))
ax1, ax2, ax3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)
ax1.plot(log['loss_train1'], label='loss_train1')
ax1.plot(log['loss_val1'], label='loss_val1')
ax1.legend(loc='upper right', prop={'size': 25})
ax1.tick_params(axis='x', labelsize='23')
ax1.tick_params(axis='y', labelsize='23')
ax2.plot(log['loss_train2'], label='loss_train2')
ax2.plot(log['loss_val2'], label='loss_val2')
ax2.legend(loc='upper right', prop={'size': 25})
ax2.tick_params(axis='x', labelsize='23')
ax2.tick_params(axis='y', labelsize='23')
ax3.plot(log['acc_train'], label='nmi_train')
ax3.plot(log['acc_val'], label='nmi_val')
ax3.legend(loc='lower right', prop={'size': 25})
ax3.tick_params(axis='x', labelsize='23')
ax3.tick_params(axis='y', labelsize='23')
plt.savefig('D:\python\GCN\DeepGraphClustering\data\experiment\log_sclump_1on2off.png')


# In[ ]:




