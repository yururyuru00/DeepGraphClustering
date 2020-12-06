import os
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

from utilities import MakePseudoLabel, ExtractAttribute, accuracy, nmi, clustering_accuracy
from models import GCN
from layers import FrobeniusNorm


loss_f = FrobeniusNorm()

def train(args, epoch, data, model, optimizer, dane_emb, log):
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
        np.save('./data/experiment/{}/Zn_epoch{}'.format(args.save_dir, epoch), Zn_np)
        
        k_means = KMeans(args.n_class, n_init=10, random_state=0, tol=0.0000001)
        k_means.fit(Zn_np)

        label = data.y.cuda().cpu().detach().numpy().copy()
        nmi = normalized_mutual_info_score(label, k_means.labels_)
        acc = clustering_accuracy(label, k_means.labels_, args.n_class)
        log['nmi'].append(nmi)
        log['acc'].append(acc)


def main():

    # setting args check
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='Cora', 
                        help='dataset of {Cora, CiteSeer, PubMed} (default: Cora)')
    parser.add_argument('-c', '--n_class', type=int, default=7,
                        help='number of class (default: 7)')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.01,
                        help='Initial learning rate. (default: 0.01)')
    parser.add_argument('-w', '--weight_decay', type=float, default=5e-4,
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help='Number of epochs to train (default: 500)')
    parser.add_argument('-g', '--gcn_layer', type=int, nargs='+', default=[512, 200],
                        help='number of hidden layer of GCN')
    parser.add_argument('-t', '--tree_depth', type=int, default=10,
                        help='tree depth of decision tree for hit idx (default: 10)')
    parser.add_argument('-s', '--save_dir', type=str, default='test',
                        help='dir name when save log (default: test)')
    args = parser.parse_args()

    # Load data
    os.makedirs('./data/experiment/{}'.format(args.save_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='./data/experiment/', name=args.dataset, 
                        pre_transform=MakePseudoLabel(args.n_class),
                        transform=ExtractAttribute(args.n_class, args.tree_depth))
    data = dataset[0].to(device)
    print(data)
    dane_emb = np.loadtxt('./data/{}/dane_emb.csv'.format(args.dataset))
    dane_emb = torch.FloatTensor(dane_emb).to(device)

    # Model and optimizer
    n_attributes = data.x.shape[1]
    model = GCN(n_attributes, args.gcn_layer).to(device)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate, 
                            weight_decay=args.weight_decay)

    # Train and Save model
    log = {'loss': [], 'nmi': []}
    for epoch in tqdm(range(args.epochs+1)):
        train(args, epoch, data, model, optimizer, dane_emb, log)
    torch.save(model.state_dict(), './data/experiment/{}/pretrained_gcn'
                .format(args.save_dir))


    # plot log
    fig = plt.figure(figsize=(35, 35))
    ax1, ax2, ax3 = fig.add_subplot(3, 1, 1), fig.add_subplot(3, 1, 2), fig.add_subplot(3, 1, 3)
    ax1.plot(log['loss'], label='loss')
    ax1.legend(loc='upper right', prop={'size': 30})
    ax1.tick_params(axis='x', labelsize='23')
    ax1.tick_params(axis='y', labelsize='23')
    ax2.plot(log['nmi'], label='nmi')
    ax2.legend(loc='upper left', prop={'size': 30})
    ax2.tick_params(axis='x', labelsize='23')
    ax2.tick_params(axis='y', labelsize='23')
    ax3.plot(log['acc'], label='acc')
    ax3.legend(loc='lower left', prop={'size': 30})
    ax3.tick_params(axis='x', labelsize='23')
    ax3.tick_params(axis='y', labelsize='23')
    plt.savefig('./data/experiment/{}/result.png'.format(args.save_dir))

    with open('./data/experiment/{}/parameters.txt'.format(args.save_dir), 'w') as w:
        for parameter in vars(args):
            w.write('{}: {}\n'.format(parameter, getattr(args, parameter)))


if __name__ == "__main__":
    main()