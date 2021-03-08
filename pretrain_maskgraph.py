import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
import torchvision.transforms as transforms
import torch.optim as optim
import os
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
import itertools

from utilities import MakePseudoLabel, ExtractAttribute, MaskGraph, BorderNodes, purity, clustering_accuracy
from models import GCN
from layers import NeuralTensorNetwork


criterion1 = torch.nn.BCEWithLogitsLoss()
criterion2 = torch.nn.BCEWithLogitsLoss()

def train(args, epoch, data, models, optimizers, log):
    model, linear_pred_nodes, linear_pred_edges = models
    optimizer, optimizer_linear_nodes, optimizer_linear_edges = optimizers

    model.train()
    linear_pred_nodes.train()
    linear_pred_edges.train()

    node_emb = model(data.masked_x, data.masked_edge_index)

    loss = torch.tensor(0.).float().to(device=data.masked_x.device.type)
    # mask the node representation
    if(args.mask_rate_node > 0.):
        pred_nodes = linear_pred_nodes(node_emb[data.masked_node_idxes])
        loss += criterion1(pred_nodes.double(), data.masked_node_label)

    # mask the edge representation
    if(args.mask_rate_edge > 0.):
        edge_preds = []
        for u, v in data.masked_edge_list:
            edge_pred = linear_pred_edges(node_emb[u], node_emb[v])
            edge_preds.append(edge_pred)
        edge_preds = torch.cat(edge_preds, axis=0)
        loss += criterion2(edge_preds, data.mask_edge_label)
    
    optimizer.zero_grad()
    optimizer_linear_nodes.zero_grad()
    optimizer_linear_edges.zero_grad()

    loss.backward()

    optimizer.step()
    optimizer_linear_nodes.step()
    optimizer_linear_edges.step()

    # logging
    log['loss'].append(loss.cuda().cpu().detach().numpy().copy())
    # when calculate clustering accuracy, we do not mask graph
    if(epoch%10 == 0):
        node_emb_not_masked = model(data.x, data.edge_index)
        Zn_np = node_emb_not_masked.cuda().cpu().detach().numpy().copy()

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
    parser.add_argument('-dp', '--drop_out', type=float, default=0.5,
                        help='drop out ratio (default: 0.5)')
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
    parser.add_argument('-mn', '--mask_rate_node', type=float, default=0.15,
                        help='mask nodes ratio (default: 0.15)')
    parser.add_argument('-me', '--mask_rate_edge', type=float, default=0.15,
                        help='mask edges ratio (default: 0.15)')
    parser.add_argument('-s', '--save_dir', type=str, default='test',
                        help='dir name when save log (default: test)')
    args = parser.parse_args()


    # load and transform dataset
    os.makedirs('./data/experiment/{}'.format(args.save_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([ExtractAttribute(args.tree_depth),
                                    MaskGraph(args.mask_rate_node, args.mask_rate_edge)])
    dataset = Planetoid(root='./data/experiment/', name=args.dataset, 
                        pre_transform=MakePseudoLabel(args.n_class), transform=transform)
    data = dataset[0].to(device)
    print(data, end='\n\n')

    # set up GCN model
    n_attributes = data.masked_x.shape[1]
    model = GCN(args.model, n_attributes, args.gcn_layer, args.drop_out).to(device)
    if(args.pretrained_gcn_dir != 'None'):
        model.load_state_dict(torch.load('./data/experiment/{}/{}/pretrained_gcn'
                                            .format(args.dataset, args.pretrained_gcn_dir)))

    # below linear model predict node attribute and if edge between nodes is exist or not
    dim_embedding = args.gcn_layer[-1]
    linear_pred_nodes = torch.nn.Linear(dim_embedding, n_attributes).to(device)
    linear_pred_edges = NeuralTensorNetwork(dim_embedding, 1).to(device)

    # set up optimizer for the GNNs
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_linear_nodes = optim.Adam(
        linear_pred_nodes.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_linear_edges = optim.Adam(
        linear_pred_edges.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    models = [model, linear_pred_nodes, linear_pred_edges]
    optimizers = [optimizer, optimizer_linear_nodes, optimizer_linear_edges]

    # train
    log = {'loss': [], 'nmi': [], 'pur': [], 'acc': []}
    for epoch in tqdm(range(args.epochs+1)):
        train(args, epoch, data, models, optimizers, log)
    torch.save(model.state_dict(), './data/experiment/{}/pretrained_gcn'
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