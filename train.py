from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import math
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.optim as optim

from utilities import load_data, accuracy, nmi, purity, kmeans, remake_to_labelorder
from models import DGC, GCN
from layers import FrobeniusNorm, purity_loss

#settingargs check
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                                help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                                help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                                help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                                help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                                help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=dict, default={'gc':[512,200],
                                'affc':[64,32], 'affr':[128,64]},  help='Number of hidden units.')
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
nmi_best, Zn_best, pseudo_label = 0., None, None
labels_sclump = np.loadtxt('D:/python/GCN/DeepGraphClustering/data/experiment/SClump_label.csv')
labels_sclump = torch.LongTensor(labels_sclump).clone().to('cuda')
dane_emb = np.loadtxt('./data/experiment/DANEemb.csv')
dane_emb = torch.FloatTensor(dane_emb).cuda()

# Model and optimizer
base_model = GCN(nfeat=features.shape[1], nhid=args.hidden['gc']).cuda()
base_model.load_state_dict(torch.load('model_gcn'))
model = DGC(base=base_model, nfeat=features.shape[1], nhid=args.hidden,
                    nclass=labels.max().item()+1, dropout=args.dropout)
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
    global pseudo_label, nmi_best, Zn_best
    t = time.time()
    model.train()
    optimizer.zero_grad()
    [output1, output2], Zn = model(features, adj)
    if(epoch%50 == 0):
        pseudo_label = kmeans(Zn, torch.max(labels)+1)
    output1_ = remake_to_labelorder(output1, pseudo_label)
    loss_train1 = F.nll_loss(output1_[idx_train], pseudo_label[idx_train]) #クロスエントロピー
    loss_train2 = loss_fro(output2[idx_train], features[idx_train]) #自作損失関数
    loss_train1.backward(retain_graph=True) #更にbackwardならretain_graph = Trueにすること
    loss_train2.backward()
    optimizer.step()
    nmi_train, pur_train = nmi(output1_[idx_train], labels[idx_train]), purity(output1[idx_train], labels[idx_train])
    log['loss_train1'].append(loss_train1.cuda().cpu().detach().item())
    log['loss_train2'].append(loss_train2.cuda().cpu().detach().item())
    log['nmi_train'].append(nmi_train)
    log['pur_train'].append(pur_train)
    if(nmi_best < nmi_train):
        nmi_best = nmi_train
        Zn_best = Zn


def test(log):
    model.eval() #evalモードだとdropoutが機能しなくなる，trainモードはもちろん機能する
    [output1, output2], Zn = model(features, adj)
    nmi_test, pur_test = nmi(output1[idx_train], labels[idx_train]), purity(output1[idx_train], labels[idx_train])
    log['nmi_test'], log['pur_test'] = nmi_test, pur_test


# Train model
t_total = time.time()
log = {'loss_train1':[], 'loss_train2':[], 'nmi_train':[], 'pur_train':[], 
           'nmi_test':0, 'pur_test':0}
print('-----------------Start Training-----------------' + '\n')
for epoch in tqdm(range(args.epochs)):
    train(epoch, log)
print("-----------------Optimization Finished!-----------------")
print("Total time elapsed: {:.4f}s\n".format(time.time() - t_total))
np.savetxt('data/experiment/BestZn_kmeans50+pretrain_.csv', Zn_best)

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
plt.savefig('data/experiment/log_kmeans50+pretrain.png')