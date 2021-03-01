import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import os
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import itertools

from utilities import MakePseudoLabel, ExtractAttribute, accuracy
from models import GCN
from layers import NeuralTensorNetwork


def train(args, epoch, data, model, optimizer, idx_train, log):

    model.train()

    optimizer.zero_grad()

    node_emb = model(data.x, data.edge_index)
    pred_clusters = F.log_softmax(node_emb, dim=1)
    loss_train = F.nll_loss(pred_clusters[idx_train], data.y[idx_train])
    acc_train = accuracy(pred_clusters[idx_train], data.y[idx_train])
    loss_train.backward()

    optimizer.step()

    # logging
    log['loss'].append(loss_train.cuda().cpu().detach().numpy().copy())
    log['acc'].append(acc_train.cuda().cpu().detach().numpy().copy())
    

def test(args, epoch, data, model, optimizer, idx_test, log):
    model.eval()
    node_emb = model(data.x, data.edge_index)
    pred_clusters = F.log_softmax(node_emb, dim=1)
    acc_test = accuracy(pred_clusters[idx_test], data.y[idx_test])
    
    # logging
    log['acc_test'] = acc_test

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
                        help='dimension of hidden layers of GCN (default: 300 300 300 300)')
    parser.add_argument('-t', '--tree_depth', type=int, default=4,
                        help='tree depth of decision tree for hit idx (default: 4)')
    parser.add_argument('-s', '--save_dir', type=str, default='test',
                        help='dir name when save log (default: test)')
    args = parser.parse_args()


    # load and transform dataset
    os.makedirs('./data/experiment/{}'.format(args.save_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='./data/experiment/', name=args.dataset, 
                        pre_transform=MakePseudoLabel(args.n_class), transform=ExtractAttribute(args.tree_depth))
    data = dataset[0].to(device)
    print(data, end='\n\n')

    idx_train = torch.LongTensor(range(20*args.n_class))
    idx_val = torch.LongTensor(range(200, 500))
    idx_test = torch.LongTensor(range(500, 1500))

    # set up GCN model
    n_attributes = data.x.shape[1]
    model = GCN(args.model, n_attributes, args.gcn_layer).to(device)
    if(args.pretrained_gcn_dir != 'None'):
        model.load_state_dict(torch.load('./data/experiment/{}/{}/pretrained_gcn'
                                            .format(args.dataset, args.pretrained_gcn_dir)))

    # set up optimizer for the GNNs
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    #############
    model.train()
    node_emb = model(data.x, data.edge_index)
    node_emb_np = node_emb.cuda().cpu().detach().numpy().copy()
    k_means = KMeans(args.n_class, n_init=10, random_state=0, tol=0.0000001)
    k_means.fit(node_emb_np)
    label = data.y.cuda().cpu().detach().numpy().copy()
    nmi = normalized_mutual_info_score(label, k_means.labels_)
    print('nmi: {}'.format(nmi))


    # train
    log = {'loss': [], 'acc': [], 'acc_test': 0.}
    for epoch in tqdm(range(args.epochs+1)):
        train(args, epoch, data, model, optimizer, idx_train, log)
    torch.save(model.state_dict(), './data/experiment/{}/trained_gcn'
                .format(args.save_dir))

    test(args, epoch, data, model, optimizer, idx_test, log)


    # log
    fig = plt.figure(figsize=(35, 35))
    ax1, ax2 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)
    ax1.plot(log['loss'], label='loss')
    ax1.legend(loc='upper right', prop={'size': 30})
    ax1.tick_params(axis='x', labelsize='23')
    ax1.tick_params(axis='y', labelsize='23')
    ax2.plot(log['acc'], label='acc')
    ax2.legend(loc='lower left', prop={'size': 30})
    ax2.tick_params(axis='x', labelsize='23')
    ax2.tick_params(axis='y', labelsize='23')
    plt.savefig('./data/experiment/{}/result.png'.format(args.save_dir))

    with open('./data/experiment/{}/result.txt'.format(args.save_dir), 'w') as w:
        w.write('loss\tacc\n')
        for loss, acc in zip(log['loss'], log['acc']):
            w.write('{:.3f}\t{:.3f}\n'.format(loss, acc))
        w.write('\ntest acc: {:.3f}'.format(log['acc_test']))

    with open('./data/experiment/{}/parameters.txt'.format(args.save_dir), 'w') as w:
        for parameter in vars(args):
            w.write('{}: {}\n'.format(parameter, getattr(args, parameter)))


if __name__ == "__main__":
    main()