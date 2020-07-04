import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
import sklearn.metrics.cluster as clus

data_path = 'D:\python\GCN\DeepGraphClustering\data'

def kmeans(data, n_of_clusters):
    n_of_clusters = n_of_clusters.cuda().cpu().detach().numpy().copy()
    k_means = KMeans(n_of_clusters, n_init=10, random_state=0, tol=0.0000001)
    k_means.fit(data)
    kmeans_labels = torch.LongTensor(k_means.labels_).clone().to('cuda')
    return kmeans_labels

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="D:/python/GCN/DeepGraphClustering/data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0])) #ここでA = A+I 更に D^-1*A までしてる
    idx_train = range(2708)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj) #ここで各入力A, X, lをtensor型に変更
    idx_train = torch.LongTensor(idx_train)

    return adj, features, labels, idx_train


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def nmi(output, labels):
    preds = output.max(1)[1].type_as(labels)
    preds = preds.cuda().cpu().detach().numpy().copy()
    labels = labels.cuda().cpu().detach().numpy().copy()
    return clus.adjusted_mutual_info_score(preds, labels, "arithmetic")

def purity(output, labels):
    preds = output.max(1)[1].type_as(labels)
    preds = preds.cuda().cpu().detach().numpy().copy()
    labels = labels.cuda().cpu().detach().numpy().copy()
    usr_size = len(preds)
    clus_size = np.max(preds)+1
    clas_size = np.max(labels)+1
    
    table = np.zeros((clus_size, clas_size))
    for i in range(usr_size):
        table[preds[i]][labels[i]] += 1
    sum = 0
    for k in range(clus_size):
        sum += np.amax(table[k])
    return sum/usr_size

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)