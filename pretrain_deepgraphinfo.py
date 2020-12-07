import argparse
import os
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from utilities import MakePseudoLabel, ExtractAttribute, purity, clustering_accuracy
from models import GCN


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module): # descriminate posotive, negative pair
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)

class Infomax(nn.Module):
    def __init__(self, gcn, discriminator):
        super(Infomax, self).__init__()
        self.gcn = gcn
        self.discriminator = discriminator
        self.loss = nn.BCEWithLogitsLoss()
        self.pool = global_mean_pool


def train(args, epoch, data, model, optimizer, device, log):
    model.train()

    node_emb = model.gcn(data.x, data.edge_index)
    
    batch_idxes = torch.LongTensor(data.pseudo_label).to(device)
    summary_emb = torch.sigmoid(model.pool(node_emb, batch_idxes))

    positive_expanded_summary_emb = summary_emb[batch_idxes]
    shifted_summary_emb = summary_emb[cycle_index(len(summary_emb), 1)]
    negative_expanded_summary_emb = shifted_summary_emb[batch_idxes]

    positive_score = model.discriminator(node_emb, positive_expanded_summary_emb)
    negative_score = model.discriminator(node_emb, negative_expanded_summary_emb)      
    
    optimizer.zero_grad()
    loss = model.loss(positive_score, torch.ones_like(positive_score)) + \
                model.loss(negative_score, torch.zeros_like(negative_score))
    loss.backward()
    optimizer.step()

    # logging
    log['loss'].append(loss.cuda().cpu().detach().numpy().copy())

    Zn_np = node_emb.cuda().cpu().detach().numpy().copy()
    k_means = KMeans(args.n_class, n_init=10, random_state=0, tol=0.0000001)
    k_means.fit(Zn_np)
    label = data.y.cuda().cpu().detach().numpy().copy()

    nmi = normalized_mutual_info_score(label, k_means.labels_)
    pur = purity(label, k_means.labels_)
    log['nmi'].append(nmi)
    log['pur'].append(pur)



def main():
    # setting args check
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of GNN')
    parser.add_argument('-d', '--dataset', type=str, default='Cora', 
                        help='dataset of {Cora, CiteSeer, PubMed} (default: Cora)')
    parser.add_argument('-c', '--n_class', type=int, default=7,
                        help='number of class (default: 7)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('-w', '--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help='number of epochs to train (defalt: 500)')
    parser.add_argument('-p', '--pretrained_gcn_dir', type=str, default='None',
                        help='dir of pretrained gcn to load (Default: None)')
    parser.add_argument('-m', '--model', type=str, default='gcn',
                        help='dataset of {gcn, gin} (default: gcn)')
    parser.add_argument('-g', '--gcn_layer', type=int, nargs='+', default=[300,300,300,300],
                        help='dimension of hidden layer of GCN (default: 300 300 300 300)')
    parser.add_argument('-t', '--tree_depth', type=int, default=4,
                        help='tree depth of decision tree for hit idx (default: 4)')
    parser.add_argument('-s', '--save_dir', type=str, default='test',
                        help='dir name when save log (default: test)')
    args = parser.parse_args()

    # load and transform dataset
    os.makedirs('./data/experiment/{}'.format(args.save_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([ExtractAttribute(args.n_class, args.tree_depth)])
    dataset = Planetoid(root='./data/experiment/', name=args.dataset, 
                        pre_transform=MakePseudoLabel(args.n_class), transform=transform)
    data = dataset[0].to(device)
    print(data, end='\n\n')

    # set up GCN model and discriminator to predict mutual information
    n_attributes = data.x.shape[1]
    gcn = GCN(args.model, n_attributes, args.gcn_layer).to(device)
    if(args.pretrained_gcn_dir != 'None'):
        gcn.load_state_dict(torch.load('./data/experiment/{}/{}/pretrained_gcn'
                                            .format(args.dataset, args.pretrained_gcn_dir)))
    n_emb_dim = args.gcn_layer[-1]
    discriminator = Discriminator(n_emb_dim)
    model = Infomax(gcn, discriminator).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                            weight_decay=args.weight_decay)

    # train
    log = {'loss': [], 'nmi': [], 'pur': [], 'acc': []}
    for epoch in tqdm(range(args.epochs+1)):
        train(args, epoch, data, model, optimizer, device, log)
    torch.save(gcn.state_dict(), './data/experiment/{}/pretrained_gcn'
                .format(args.save_dir))

    # log
    fig = plt.figure(figsize=(35, 35))
    ax1, ax2, ax3 = fig.add_subplot(3, 1, 1), fig.add_subplot(3, 1, 2), fig.add_subplot(3, 1, 3)
    ax1.plot(log['loss'], label='loss')
    ax1.legend(loc='upper right', prop={'size': 30})
    ax1.tick_params(axis='x', labelsize='23')
    ax1.tick_params(axis='y', labelsize='23')
    ax2.plot(log['nmi'], label='nmi')
    ax2.legend(loc='lower left', prop={'size': 30})
    ax2.tick_params(axis='x', labelsize='23')
    ax2.tick_params(axis='y', labelsize='23')
    ax3.plot(log['pur'], label='purity')
    ax3.legend(loc='lower left', prop={'size': 30})
    ax3.tick_params(axis='x', labelsize='23')
    ax3.tick_params(axis='y', labelsize='23')
    plt.savefig('./data/experiment/{}/result.png'.format(args.save_dir))

    with open('./data/experiment/{}/result.txt'.format(args.save_dir), 'w') as w:
        w.write('loss\tnmi \tpurity\n')
        for loss, nmi, purity in zip(log['loss'], log['nmi'], log['pur']):
            w.write('{:.3f}\t{:.3f}\t{:.3f}\n'.format(loss, nmi, purity))

    with open('./data/experiment/{}/parameters.txt'.format(args.save_dir), 'w') as w:
        for parameter in vars(args):
            w.write('{}: {}\n'.format(parameter, getattr(args, parameter)))


if __name__ == "__main__":
    main()
