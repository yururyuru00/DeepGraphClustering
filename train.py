import torch
import argparse
from torch_geometric.datasets import Planetoid
from utilities import ExtractSubstructureContextPair
from models import GCN
import torch.optim as optim


def train(args, model_substruct, model_context, data, optimizer_substruct, optimizer_context, device):
    pass


parser = argparse.ArgumentParser(
    description='PyTorch implementation of pre-training of GNN')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (defalt: 100)')
parser.add_argument('--hidden', type=list, default=[1024, 768, 512, 384, 256],
                    help='number of hidden layer of GCN for substract representation')
args = parser.parse_args()

# load and transform dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='./data/experiment/', name='Cora')
data = dataset[0]
print(data, end='\n\n')

# set up GCN model
n_attributes = len(data.x[0])
model_substruct = GCN(args.w_substract, n_attributes, args.hidden1).to(device)
model_context = GCN(args.w_context, n_attributes, args.hidden2).to(device)

# set up optimizer for the two GNNs
optimizer_substruct = optim.Adam(
    model_substruct.parameters(), lr=args.lr, weight_decay=args.decay)
optimizer_context = optim.Adam(
    model_context.parameters(), lr=args.lr, weight_decay=args.decay)

# train
for epoch in range(args.epochs):
    print("====epoch " + str(epoch))

    train(args, model_substruct, model_context, data.to(device),
          optimizer_substruct, optimizer_context, device)
