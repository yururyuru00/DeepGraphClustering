#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans

data_path = 'D:\python\GCN\DeepGraphClustering\data'

def kmeans(data, n_of_clusters):
    n_of_clusters = n_of_clusters.cuda().cpu().detach().numpy().copy()
    k_means = KMeans(n_of_clusters, n_init=10, tol=0.0000001)
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

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj) #ここで各入力A, X, lをtensor型に変更
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


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


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# In[ ]:


'''
import re
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sp 
import glob

dataset = 'D:\python\GCN\DeepGraphClustering\data\MUTAG'
n_of_labels = 7
batch_size = 3
n_of_nodes = np.zeros(batch_size, dtype=int)

def encode_onehot(val):
    return np.array([1 if i==val else 0 for i in range(n_of_labels)])

def size_loader(ls):
    labels = []
    for i, l in enumerate(ls):
        if(l == '\n'):
            break
    return i/4

def data_loader(ls):
    labels = []
    for i, l in enumerate(ls):
        if(l == '\n'):
            break
    nodes = [ls[j] for j in range(i)]
    edges = [ls[k] for k in range(i+1, len(ls))]

    map_ = {}
    for i, n in enumerate(nodes):
        if(i%4==1):
            idx = int(re.findall(r'id (\d+)', n)[0])
            map_[idx] = int((i+3)/4)-1
        if(i%4==2):
            n = n.replace('"', '')
            n = int(re.findall(r'label (\d)', n)[0])
            labels.append(n)
    
    pairs = []
    for i, e in enumerate(edges):
        if(i%5==1):
            source = int(re.findall(r'source (\d+)', e)[0])
        if(i%5==2):
            target = int(re.findall(r'target (\d+)', e)[0])
            pairs.append((map_[source], map_[target]))

    X = np.array([encode_onehot(l) for i, l in enumerate(labels)])
    A = np.zeros((len(labels), len(labels)), dtype=int)
    for i, j in pairs:
        A[i][j] = 1
    return csr_matrix(X), coo_matrix(A)

with open(dataset + r'\0.gml', 'r') as f:
    ls = f.readlines()
    ls = ls[2:-2]
    
Xs, As = [], []
gmls = glob.glob('D:\python\GCN\DeepGraphClustering\data\MUTAG\*.gml')
for i in range(batch_size):
    with open(gmls[i], 'r') as f:
        ls = f.readlines()
        ls = ls[2:-2]
    n_of_nodes[i] = size_loader(ls)

A = csr_matrix(([], ([], [])), shape=(0, np.sum(n_of_nodes)), dtype=int)
X = csr_matrix(([], ([], [])), shape=(0, n_of_labels), dtype=int)
for i in range(batch_size):
    with open(gmls[i], 'r') as f:
        ls = f.readlines()
        ls = ls[2:-2]
    x, a = data_loader(ls)
    print(x.toarray(), end='\n\n')
    X = sp.vstack((X, x), format='csr')
    Ai = csr_matrix(([], ([], [])), shape=(n_of_nodes[i], 0), dtype=int)
    for j in range(batch_size):
        if(i==j):
            Ai = sp.hstack((Ai, a), format='csr')
        else:
            x = csr_matrix(([], ([], [])), shape=(n_of_nodes[i], n_of_nodes[j]), dtype=int)
            Ai = sp.hstack((Ai, x), format='csr')
    A = sp.vstack((A, Ai), format='csr')

X, A = X.toarray(), A.toarray()
np.savetxt('D:\python\GCN\DeepGraphClustering\data\X.csv', X, fmt='%d')
np.savetxt('D:\python\GCN\DeepGraphClustering\data\A.csv', A, fmt='%d')

with open(dataset + r'\Labels.txt', 'r') as f:
    classes = np.array([int(c.strip()) for c in f.readlines()])
'''


# In[ ]:




