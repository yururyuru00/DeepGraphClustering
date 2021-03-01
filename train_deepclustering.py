from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import argparse
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from utilities import ExtractAttribute, remake_to_labelorder, nmi, purity
from layers import FrobeniusNorm
from models import DGC, GCN
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from skfuzzy.cluster import cmeans
import math
import time
from kmeans_pytorch import kmeans


loss_frobenius = FrobeniusNorm()
global pseudo_label

def train(args, epoch, model_dgc, data, optimizer, log):
    global pseudo_label

    model_dgc.train()
    [pred_clusters, reconstructed_x], Zn = model_dgc(data.x, data.edge_index)

    # make pseudo labels by using k-means
    Zn = Zn.cuda().cpu().detach().numpy().copy()
    result = cmeans(Zn.T, args.n_class, 1.1, 0.001, 10000, seed=0)
    fuzzy_means = result[1].T
    pseudo_label = torch.FloatTensor(np.array(fuzzy_means)).cuda()
    pred_clusters = remake_to_labelorder(pred_clusters, pseudo_label)
    
    loss_clustering = loss_frobenius(pred_clusters, pseudo_label)
    loss_reconstruct = loss_frobenius(reconstructed_x, data.x)
    loss = loss_clustering

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # loging every time
    pred_clusters = pred_clusters.cuda().cpu().detach().numpy().copy()
    pred_hard_clusters = np.array([np.argmax(i) for i in pred_clusters])
    labels = data.y.cuda().cpu().detach().numpy().copy()
    nmi = normalized_mutual_info_score(labels, pred_hard_clusters)
    pur = purity(labels, pred_hard_clusters)

    log['loss_clus'].append(loss_clustering.cuda().cpu().detach().numpy().copy())
    log['loss_reconst'].append(loss_reconstruct.cuda().cpu().detach().numpy().copy())
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
    parser.add_argument('-r', '--rate_dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help='number of epochs to train (defalt: 500)')
    parser.add_argument('-p', '--pretrained_gcn_dir', type=str, default='test',
                        help='dir of pretrained gcn to load (Default: test)')
    parser.add_argument('-m', '--model', type=str, default='gcn',
                        help='dataset of {gcn, gin} (default: gcn)')
    parser.add_argument('-g', '--gcn_layer', type=int, nargs='+', default=[128, 64, 32, 16],
                        help='number of hidden layer of GCN (default: 128 64 32 16)')
    parser.add_argument('-cl', '--clustering_layer', type=int, nargs='+', default=[64, 32],
                        help='number of hidden layer of clustering MLP (default: 64 32)')
    parser.add_argument('-rl', '--reconstruct_layer', type=int, nargs='+', default=[128, 64],
                        help='number of hidden layer of reconstruct MLP (default: 128 64)')
    parser.add_argument('-t', '--tree_depth', type=int, default=4,
                            help='tree depth of decision tree for hit idx (default: 4)')
    parser.add_argument('-s', '--save_dir', type=str, default='test',
                        help='dir name when save log (default: test)')
    args = parser.parse_args()

    # load and transform dataset
    os.makedirs('./data/experiment/{}'.format(args.save_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='./data/experiment/', name=args.dataset, 
                        transform=ExtractAttribute(args.n_class, args.tree_depth))
    data = dataset[0].to(device)
    print(data, end='\n\n')

    # set up DGC model
    n_attributes = data.x.shape[1]
    n_class = torch.max(data.y).cuda().cpu().detach().numpy().copy()+1
    base_model = GCN(args.model, n_attributes, args.gcn_layer).to(device)
    base_model.load_state_dict(torch.load('./data/experiment/{}/{}/pretrained_gcn'
                                            .format(args.dataset, args.pretrained_gcn_dir)))
    num_hidden = {'gcn': args.gcn_layer, 'clustering': args.clustering_layer, 
                    'reconstruct': args.reconstruct_layer}
    model_dgc = DGC(base_model, n_attributes, num_hidden, args.n_class, args.rate_dropout).to(device)

    # set up optimizer
    optimizer = optim.Adam(model_dgc.parameters(),
                           lr=args.learning_rate, weight_decay=args.weight_decay)

    # train
    log = {'loss_clus': [], 'loss_reconst': [], 'nmi': [], 'pur': []}
    for epoch in tqdm(range(args.epochs+1)):
        train(args, epoch, model_dgc, data, optimizer, log)


    # log
    fig = plt.figure(figsize=(35, 35))
    ax1, ax2, ax3, ax4 = fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2), \
        fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)
    ax1.plot(log['loss_clus'], label='loss_clustering')
    ax1.legend(loc='upper right', prop={'size': 25})
    ax1.tick_params(axis='x', labelsize='23')
    ax1.tick_params(axis='y', labelsize='23')
    ax2.plot(log['loss_reconst'], label='loss_reconstruct')
    ax2.legend(loc='upper right', prop={'size': 25})
    ax2.tick_params(axis='x', labelsize='23')
    ax2.tick_params(axis='y', labelsize='23')
    ax3.plot(log['nmi'], label='nmi')
    ax3.legend(loc='lower right', prop={'size': 25})
    ax3.tick_params(axis='x', labelsize='23')
    ax3.tick_params(axis='y', labelsize='23')
    ax3.set_ylim(min(log['nmi']), math.ceil(10*max(log['nmi']))/10)
    ax4.plot(log['pur'], label='purity')
    ax4.legend(loc='lower right', prop={'size': 25})
    ax4.tick_params(axis='x', labelsize='23')
    ax4.tick_params(axis='y', labelsize='23')
    ax4.set_ylim(min(log['pur']), math.ceil(10*max(log['pur']))/10)
    plt.savefig('./data/experiment/{}/result.png'.format(args.save_dir))

    with open('./data/experiment/{}/parameters.txt'.format(args.save_dir), 'w') as w:
            for parameter in vars(args):
                w.write('{}: {}\n'.format(parameter, getattr(args, parameter)))
        
    with open('./data/experiment/{}/result.txt'.format(args.save_dir), 'w') as w:
        w.write('loss\tnmi \tpurity\n')
        for loss, nmi, purity in zip(log['loss_clus'], log['nmi'], log['pur']):
            w.write('{:.3f}\t{:.3f}\t{:.3f}\n'.format(loss, nmi, purity))


if __name__ == "__main__":
    main()