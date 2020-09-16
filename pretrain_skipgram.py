import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import KarateClub
import torch.optim as optim

from utilities import ExtractSubstructureContextPair
from models import GCN
from debug import plot_Zn


def train(epoch, model_substruct, model_context, data, optimizer_substruct, optimizer_context):
    model_substruct.train()
    model_context.train()

    # creating substract, context, and negative representations
    representations = model_substruct(
        data.x_substruct, data.edge_index_substruct)
    substruct_rep = representations[data.center_substruct_idx]
    context_rep = model_context(data.x_context, data.edge_index_context)[
        data.overlap_context_substruct_idx]
    negative_rep = model_substruct(data.x_substruct, data.edge_index_substruct)[
        data.center_negative_idx].reshape(1, -1)

    Zn = representations.cuda().cpu().detach().numpy().copy()
    label = data.y.cuda().cpu().detach().numpy().copy()
    plot_Zn(
        Zn, label, path_save='./data/experiment/pretrain_skipgram/Zn_skipgram_epoch{}'.format(epoch))

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
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (defalt: 100)')
parser.add_argument('--border', type=int, default=1,
                    help='boderline between substract and context graph (default: 3).')
parser.add_argument('--w_substract', type=int, default=5,
                    help='width of substruct graph (default: 5).')
parser.add_argument('--w_context', type=int, default=3,
                    help='width of context graph (default: 3).')
parser.add_argument('--hidden1', type=list, default=[1024, 768, 512, 384, 256],
                    help='number of hidden layer of GCN for substract representation')
parser.add_argument('--hidden2', type=list, default=[1024, 512, 256],
                    help='number of hidden layer of GCN for context representation')
args = parser.parse_args()

# load and transform dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='./data/experiment/', name='Cora',
                    transform=ExtractSubstructureContextPair(args.border))
data = dataset[0]

# set up GCN model
n_attributes = data.x.shape[1]
model_substruct = GCN(args.w_substract, n_attributes, args.hidden1).to(device)
model_context = GCN(args.w_context, n_attributes, args.hidden2).to(device)

# set up optimizer for the two GNNs
optimizer_substruct = optim.Adam(
    model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
optimizer_context = optim.Adam(
    model_context.parameters(), lr=args.lr, weight_decay=args.decay)

# train
log = []
for epoch in tqdm(range(args.epochs)):
    loss = train(epoch, model_substruct, model_context, dataset[0].to(device),
                 optimizer_substruct, optimizer_context)
    log.append(loss)
torch.save(model_substruct.state_dict(), 'pretrained_gcn')

# log
fig = plt.figure(figsize=(17, 17))
plt.plot(log, label='cross entropy loss')
plt.legend(loc='upper right', prop={'size': 12})
plt.tick_params(axis='x', labelsize='12')
plt.tick_params(axis='y', labelsize='12')
plt.savefig('./data/experiment/pretrain_skipgram/loss.png')
