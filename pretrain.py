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

#settingargs check
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=dict, default={"gc":[512,200], 
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
adj, features, labels, idx_train = load_data()
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

if args.cuda: #cpuかgpuどちらのtensorを使うかの処理
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()

def train(epoch, log):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    [output1, output2], Zn = model(features, adj)
    '''if(epoch%100==0 and np.random.rand() <= epoch/args.epochs): #epoch/epochsの確率でkmeansする
        label = kmeans(Zn, torch.max(labels)+1)
    else:
        label = labels_sclump'''
    label = labels_sclump
    #labelの順序が変わるからおかしくなる
    loss_train1 = F.nll_loss(output1[idx_train], label[idx_train]) #クロスエントロピー
    loss_train2 = loss_fro(output2[idx_train], features[idx_train]) #自作損失関数
    loss_train1.backward() #更にbackwardならretain_graph = Trueにすること
    #loss_train2.backward()
    optimizer.step()
    nmi_train, pur_train = nmi(output1[idx_train], labels[idx_train]), purity(output1[idx_train], labels[idx_train])
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    
    '''if not args.fastmode: #defaltはFalseなのでここの処理は行う
        # deactivates dropout during validation run.
        model.eval()
        [output1, output2], Zn = model(features, adj)  
    #kmeans_labels = kmeans(Zn, torch.max(labels)+1)
    loss_val1 = F.nll_loss(output1[idx_val], labels_sclump[idx_val])
    loss_val2 = loss_fro(output2[idx_val], features[idx_val])
    acc_val = nmi(output1[idx_val], labels[idx_val]) #acc_valはnmiの場合，npで返ってくる'''
    
    log['loss_train1'].append(loss_train1.cuda().cpu().detach().numpy().copy())
    log['loss_train2'].append(loss_train2.cuda().cpu().detach().numpy().copy()/len(idx_train))
    log['nmi_train'].append(nmi_train)
    log['pur_train'].append(pur_train)
    '''log['loss_val1'].append(loss_val1.cuda().cpu().detach().numpy().copy())
    log['loss_val2'].append(loss_val2.cuda().cpu().detach().numpy().copy()/len(idx_val))
    log['acc_val'].append(acc_val)'''


def test(log):
    model.eval() #evalモードだとdropoutが機能しなくなる，trainモードはもちろん機能する
    [output1, output2], Zn = model(features, adj)
    np.savetxt('D:\python\GCN\DeepGraphClustering\data\experiment\Zn_sclump_1on2off.csv', Zn)
    nmi_test, pur_test = nmi(output1[idx_train], labels[idx_train]), purity(output1[idx_train], labels[idx_train])
    log['nmi_test'], log['pur_test'] = nmi_test, pur_test


# Train model
t_total = time.time()
log = {'loss_train1':[], 'loss_train2':[], 'nmi_train':[], 'pur_train':[], 
       'nmi_test':0, 'pur_test':0}
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
          'nmi_train: {:.4f}'.format(log['nmi_train'][epoch]),
          'pur_train: {:.4f}'.format(log['pur_train'][epoch]))
print("#################\nTest set results:", "nmi= {:.4f}".format(log['nmi_test']))
fig = plt.figure(figsize=(35, 35))
ax1, ax2, ax3, ax4 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2),                     fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)
ax1.plot(log['loss_train1'], label='loss_train1')
ax1.legend(loc='upper right', prop={'size': 25})
ax1.tick_params(axis='x', labelsize='23')
ax1.tick_params(axis='y', labelsize='23')
ax2.plot(log['loss_train2'], label='loss_train2')
ax2.legend(loc='upper right', prop={'size': 25})
ax2.tick_params(axis='x', labelsize='23')
ax2.tick_params(axis='y', labelsize='23')
ax3.plot(log['nmi_train'], label='nmi_train')
ax3.plot(np.array([log['nmi_test'] for _ in range(args.epochs)]), label='nmi_test')
ax3.legend(loc='lower right', prop={'size': 25})
ax3.tick_params(axis='x', labelsize='23')
ax3.tick_params(axis='y', labelsize='23')
ax3.set_ylim(min(log['nmi_train']), math.ceil(10*max(log['nmi_train']))/10)
ax4.plot(log['pur_train'], label='pur_train')
ax4.plot(np.array([log['pur_test'] for _ in range(args.epochs)]), label='pur_test')
ax4.legend(loc='lower right', prop={'size': 25})
ax4.tick_params(axis='x', labelsize='23')
ax4.tick_params(axis='y', labelsize='23')
ax4.set_ylim(min(log['pur_train']), math.ceil(10*max(log['pur_train']))/10)
plt.savefig('D:\python\GCN\DeepGraphClustering\data\experiment\log_sclump_1on2off.png')