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

from utilities import ExtractAttribute, ExtractSubstructureContextPair, nmi
from models import GCN
from debug import plot_Zn


def train(args, epoch, data, models, optimizers, log):
    model_substruct, model_context = models
    optimizer_substruct, optimizer_context = optimizers

    model_substruct.train()
    model_context.train()

    n_class = len(data.list)
    class_labels = np.arange(n_class)
    map_postitive_negative_idx = np.roll(class_labels, 1)
    for c_i in range(n_class):

        # creating substract, context, and negative representations
        representations = model_substruct(
            data.x, data.edge_index)
        substruct_rep = representations[data.list[c_i].center_idx]
        negative_rep = representations[data.list[map_postitive_negative_idx[c_i]]
                                        .center_idx].reshape(1, -1)
        context_rep = model_context(data.list[c_i].x_context, data.list[c_i].edge_index_context) \
                                    [data.list[c_i].context_idxes]

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

        # logging and debug
        log['loss'].append(loss.cuda().cpu().detach().numpy().copy())
        if(epoch % 10 == 0 and c_i == 0):  
            Zn_np = representations.cuda().cpu().detach().numpy().copy()
            np.save('./data/experiment/{}/Zn_epoch{}'.format(args.save_dir, epoch), Zn_np)

            k_means = KMeans(args.n_class, n_init=10, random_state=0, tol=0.0000001)
            k_means.fit(Zn_np)

            label = data.y.cuda().cpu().detach().numpy().copy()
            nmi = normalized_mutual_info_score(label, k_means.labels_)
            log['nmi'].append(nmi)


def main():
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of pre-training of GNN')
    parser.add_argument('--dataset', type=str, default='Cora', 
                        help='dataset of {Cora, Citeseer, Pubmed} (default: Cora)')
    parser.add_argument('--n_class', type=int, default=7,
                        help='number of class')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs to train (defalt: 500)')
    parser.add_argument('--hidden1', type=int, nargs='+', default=[128, 64, 32],
                        help='number of hidden layer of substruct-GCN (default: 128 64 32)')
    parser.add_argument('--hidden2', type=int, nargs='+', default=[128, 32],
                        help='number of hidden layer of context-GCN (default: 128 32)')
    parser.add_argument('--tree_depth', type=int, default=10,
                        help='tree depth of decision tree for hit idx (default: 10)')
    parser.add_argument('--border', type=int, default=1,
                        help='boderline between substract and context graph (default: 3).')
    parser.add_argument('--save_dir', type=str, default='test',
                        help='dir name when save log (default: test)')
    args = parser.parse_args()

    # load and transform dataset
    os.makedirs('./data/experiment/{}'.format(args.save_dir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([ExtractAttribute(args.n_class, args.tree_depth),
                                    ExtractSubstructureContextPair(args.n_class, args.border, device)])
    dataset = Planetoid(root='./data/experiment/', name=args.dataset, transform=transform)
    data = dataset[0].to(device)
    print(data)

    # set up GCN model
    n_attributes = data.x.shape[1]
    model_substruct = GCN(n_attributes, args.hidden1).to(device)
    model_context = GCN(n_attributes, args.hidden2).to(device)

    # set up optimizer for the two GNNs
    optimizer_substruct = optim.Adam(
        model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_context = optim.Adam(
        model_context.parameters(), lr=args.lr, weight_decay=args.decay)
    
    models = [model_substruct, model_context]
    optimizers = [optimizer_substruct, optimizer_context]

    # train
    log = {'loss': [], 'nmi': []}
    for epoch in tqdm(range(args.epochs)):
        train(args, epoch, data, models, optimizers, log)
    torch.save(model.state_dict(), './data/experiment/{}/pretrained_gcn'
                .format(args.args.save_dir))

    # log
    fig = plt.figure(figsize=(35, 35))
    ax1, ax2 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)
    ax1.plot(log['loss'], label='loss')
    ax1.legend(loc='upper right', prop={'size': 25})
    ax1.tick_params(axis='x', labelsize='23')
    ax1.tick_params(axis='y', labelsize='23')
    ax2.plot(log['nmi'], label='nmi')
    ax2.legend(loc='upper left', prop={'size': 25})
    ax2.tick_params(axis='x', labelsize='23')
    ax2.tick_params(axis='y', labelsize='23')
    plt.savefig('./data/experiment/{}/result.png'.format(args.save_dir))

    with open('./data/experiment/{}/parameters.txt'.format(args.save_dir), 'w') as w:
        for parameter in vars(args):
            w.write('{}: {}\n'.format(parameter, getattr(args, parameter)))


if __name__ == "__main__":
    main()
