from __future__ import division
from layers import FrobeniusNorm, HardClusterLoss
from models import DGC, GCN
from utilities import *
import torch.optim as optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from __future__ import print_function

from tqdm import tqdm
import os
import math
import sys
sys.path.append('D:/python/GCN/DeepGraphClustering/data/utilities')


# settingargs check
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1000, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--skips', type=int, default=50,
                    help='Number of epochs per feed preudo labels.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=dict, default={'gc': [512, 200],
                                                    'affc': [64, 32], 'affr': [128, 64]},  help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save', type=str, default='test',
                    help='filename when save log')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
tracedir = '{}_epoch{}_skip{}'.format(args.save, args.epochs, args.skips)
os.makedirs('./data/experiment/' + tracedir)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, _ = load_data()
pseudo_label = None
# Model and optimizer
base_model = GCN(nfeat=features.shape[1], nhid=args.hidden['gc']).cuda()
base_model.load_state_dict(torch.load('model_gcn_mini'))
model = DGC(base=base_model, nfeat=features.shape[1], nhid=args.hidden,
            nclass=labels.max().item()+1, dropout=args.dropout)
loss_fro = FrobeniusNorm()
loss_hardcluster = HardClusterLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)  # lrが学習係数
if args.cuda:  # enable or disable cuda(GPU)
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()

# train function


def train(epoch, log):
    global pseudo_label, nmi_best, Zn_best
    t = time.time()
    model.train()
    optimizer.zero_grad()
    [output1, output2], Zn = model(features, adj)
    if(epoch % args.skips == 0):
        m_setting = 1.1 + 0.1*(float(epoch)/float(args.epochs))
        pseudo_label = fuzzy_cmeans(Zn, torch.max(labels)+1, m=m_setting)
    output1_mapped = remake_to_labelorder(output1, pseudo_label)
    loss_train1 = loss_fro(output1_mapped, pseudo_label)  # クロスエントロピー
    #loss_train1 += loss_hardcluster(output1_mapped)
    loss_train2 = loss_fro(output2, features)
    # 更にbackwardならretain_graph = Trueにすること
    loss_train1.backward(retain_graph=True)
    loss_train2.backward()
    optimizer.step()
    nmi_train, pur_train = nmi(output1, labels), purity(output1, labels)

    # loging every time
    log['loss_train1'].append(loss_train1.item())
    log['loss_train2'].append(loss_train2.item())
    log['nmi_train'].append(nmi_train)
    log['pur_train'].append(pur_train)
    pseudo_label_ = pseudo_label.cuda().cpu().detach().numpy().copy()
    output1_ = output1.cuda().cpu().detach().numpy().copy()
    output1_mapped_ = output1_mapped.cuda().cpu().detach().numpy().copy()
    np.save('./data/experiment/' + tracedir + '/Zn_epoch#{}'.format(epoch), Zn)
    np.save('./data/experiment/' + tracedir +
            '/pseudolabel_epoch#{}'.format(epoch), pseudo_label_)
    np.save('./data/experiment/' + tracedir +
            '/predlabel_epoch#{}'.format(epoch), output1_)
    np.save('./data/experiment/' + tracedir +
            '/predlabel_mapped_epoch#{}'.format(epoch), output1_mapped_)


def test(log):
    model.eval()  # evalモードだとdropoutが機能しなくなる，trainモード時は機能する
    [output1, output2], Zn = model(features, adj)
    nmi_test, pur_test = nmi(output1, labels), purity(output1, labels)
    log['nmi_test'], log['pur_test'] = nmi_test, pur_test


# Train
t_total = time.time()
log = {'loss_train1': [], 'loss_train2': [], 'nmi_train': [], 'pur_train': [],
       'nmi_test': 0, 'pur_test': 0}
print('-----------------Start Training-----------------' + '\n')
for epoch in tqdm(range(args.epochs)):
    train(epoch, log)
print("-----------------Optimization Finished!-----------------")
print("Total time elapsed: {:.4f}s\n".format(time.time() - t_total))

# Test
test(log)


#log + plot
with open('./data/experiment/' + tracedir + '/result.csv', 'w') as w:
    w.write('Epoch lss1_train lss2_train nmi_train pur_train\n')
    for epoch in range(args.epochs):
        w.write('{} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(epoch, log['loss_train1'][epoch],
                                                          log['loss_train2'][epoch], log['nmi_train'][epoch], log['pur_train'][epoch]))

fig = plt.figure(figsize=(35, 35))
ax1, ax2, ax3, ax4 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), \
    fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)
ax1.plot(log['loss_train1'], label='loss_train1')
ax1.legend(loc='upper right', prop={'size': 25})
ax1.tick_params(axis='x', labelsize='23')
ax1.tick_params(axis='y', labelsize='23')
ax2.plot(log['loss_train2'], label='loss_train2')
ax2.legend(loc='upper right', prop={'size': 25})
ax2.tick_params(axis='x', labelsize='23')
ax2.tick_params(axis='y', labelsize='23')
ax3.plot(log['nmi_train'], label='nmi_train')
ax3.plot(np.array([log['nmi_test']
                   for _ in range(args.epochs)]), label='nmi_test')
ax3.legend(loc='lower right', prop={'size': 25})
ax3.tick_params(axis='x', labelsize='23')
ax3.tick_params(axis='y', labelsize='23')
ax3.set_ylim(min(log['nmi_train']), math.ceil(10*max(log['nmi_train']))/10)
ax4.plot(log['pur_train'], label='pur_train')
ax4.plot(np.array([log['pur_test']
                   for _ in range(args.epochs)]), label='pur_test')
ax4.legend(loc='lower right', prop={'size': 25})
ax4.tick_params(axis='x', labelsize='23')
ax4.tick_params(axis='y', labelsize='23')
ax4.set_ylim(min(log['pur_train']), math.ceil(10*max(log['pur_train']))/10)
plt.savefig('./data/experiment/' + tracedir + './result.png')
