from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import argparse
from torch_geometric.datasets import KarateClub
import torch.nn.functional as F
from utilities import remake_to_labelorder, kmeans, nmi, purity
from layers import FrobeniusNorm
from models import DGC, GCN
import torch.optim as optim
from sklearn.cluster import KMeans


loss_frobenius = FrobeniusNorm()


def train(model_dgc, data, optimizer, log):
    model_dgc.train()
    optimizer.zero_grad()
    [clus_labels, reconstructed_x], Zn = model_dgc(data.x, data.edge_index)

    # make pseudo labels by using k-means
    Zn_np = Zn.cuda().cpu().detach().numpy().copy()
    n_class = torch.max(data.y).cuda().cpu().detach().numpy().copy() + 1
    k_means = KMeans(n_class, n_init=10, random_state=0, tol=0.0000001)
    k_means.fit(Zn_np)
    pseudo_label = torch.LongTensor(k_means.labels_).cuda()

    # map the two labels (predicted labels and pseudo labels) to each other.
    clus_labels_mapped = remake_to_labelorder(clus_labels, pseudo_label)

    loss_clustering = F.nll_loss(
        clus_labels_mapped, pseudo_label)
    loss_reconstruct = loss_frobenius(reconstructed_x, data.x)
    loss = loss_clustering + loss_reconstruct
    loss.backward()
    optimizer.step()

    nmi_train = nmi(clus_labels, data.y)
    pur_train = purity(clus_labels, data.y)

    # loging every time
    log['loss_clustering'].append(loss_clustering.item())
    log['loss_reconstruct'].append(loss_reconstruct.item())
    log['nmi'].append(nmi_train)
    log['pur'].append(pur_train)


parser = argparse.ArgumentParser(
    description='PyTorch implementation of pre-training of GNN')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (defalt: 100)')
args = parser.parse_args()

# load and transform dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = KarateClub(transform=None)
data = dataset[0]
print(data, end='\n\n')

# set up DGC model
n_attributes = data.x.shape[1]
n_class = torch.max(data.y).cuda().cpu().detach().numpy().copy()+1
hidden = {'gc': [1024, 768, 512, 384, 256],
          'clustering': [64, 32], 'reconstruct': [128, 64]}
base_model = GCN(n_layer=5, n_feat=n_attributes, hid=hidden['gc']).to(device)
base_model.load_state_dict(torch.load('./pretrained_gcn'))
model_dgc = DGC(base=base_model, n_feat=n_attributes, n_hid=hidden,
                n_class=n_class, dropout=args.dropout).to(device)

# set up optimizer
optimizer = optim.Adam(model_dgc.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# train
log = {'loss_clustering': [], 'loss_reconstruct': [], 'nmi': [], 'pur': []}
for epoch in tqdm(range(args.epochs)):
    train(model_dgc, data.to(device), optimizer, log)


# log
fig = plt.figure(figsize=(17, 17))
plt.plot(log['loss_clustering'], label='loss clustering')
plt.legend(loc='upper right', prop={'size': 12})
plt.tick_params(axis='x', labelsize='12')
plt.tick_params(axis='y', labelsize='12')
plt.show()
