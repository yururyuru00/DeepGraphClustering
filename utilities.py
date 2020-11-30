import time
import random
import copy
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from skfuzzy.cluster import cmeans
import sklearn.metrics.cluster as clus
from sklearn.metrics import accuracy_score
import itertools
import networkx as nx
from torch_geometric.data import Data
from torch_geometric import utils
from scipy.sparse.linalg import eigsh
from scipy import linalg

from debug import plot_G_contextG_pair


class AugumentGraph:

    def __init__(self, num_steps, num_edges_BA, num_edges_TF):
        self.num_steps = num_steps
        self.num_edges_BA = num_edges_BA
        self.num_edges_TF = num_edges_TF
    
    def __call__(self, data):
        num_nodes = data.x.size()[0]
        num_dims = data.x.size()[1]
        degree = [0 for w in range(num_nodes)]

        for i,j in data.edge_index.T:
                degree[i] += 1
                degree[j] += 1

        for step in range(self.num_steps):

            # add new node
            new_node_attr = torch.zeros(1, num_dims, dtype=torch.float)
            data.y = torch.cat((data.y, torch.LongTensor([7])), 
                                dim=0) # label 7 means augmented vertices
            num_nodes += 1
            new_node_id = num_nodes-1
            neighbor_node_idxes_of_newnode = []

            # select nodes which are connected to new node (BA algorithm)
            degree_buff = degree.copy()
            w_list = []
            for tri in range(self.num_edges_BA):
                w = random.choices(range(num_nodes-1), weights=degree_buff)[0]
                w_list.append(w)
                degree_buff[w] = 0
            degree.append(0) # append new node's degree

            for w in w_list:

                # Barabasi Albert (BA) Algorithm
                data.edge_index = torch.cat((data.edge_index, torch.tensor([[new_node_id], [w]])), dim=1)
                degree[w] += 1
                degree[new_node_id] += 1
                neighbor_node_idxes_of_newnode.append(w)

                # Triad Formation (TF) Algorithm   
                u_list = sample_from_neighbors(data, w, self.num_edges_TF)
                for u in u_list:
                    data.edge_index = torch.cat((data.edge_index, torch.tensor([[new_node_id], [u]])), dim=1)
                    degree[u] += 1
                    degree[new_node_id] += 1
                    neighbor_node_idxes_of_newnode.append(u)

            # edit new node's attribute based on neighbor nodes
            for idx in neighbor_node_idxes_of_newnode:
                new_node_attr = torch.cat((new_node_attr, data.x[idx].view(1, -1)), dim=0)
            new_node_attr = torch.mean(new_node_attr, dim=0, keepdim=True)
            new_node_attr = torch.round(new_node_attr)
            data.x = torch.cat((data.x, new_node_attr), dim=0)

        return data             

def sample_from_neighbors(data, i, sample_size):
    A = utils.to_dense_adj(data.edge_index)[0].cpu().numpy()
    neighbors_list = [neighbor for neighbor in np.where(A[i]==1)[0]]
    sample_size = min(len(neighbors_list), sample_size)
    return random.sample(neighbors_list, sample_size)

class MakePseudoLabel:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def normalized(self, mat):
        for i in range(len(mat)):
            sum_ = np.sum(mat[i])
            for j in range(len(mat[i])):
                if(sum_ == 0):
                    mat[i][j] = 0.
                else:
                    mat[i][j] = mat[i][j] / sum_

    def makeKnn(self, mat, k):
        knnmat = np.zeros((len(mat), len(mat[0])))
        for i in range(len(mat)):
            arg = np.argsort(-mat[i])
            for top in range(k):
                knnmat[i][arg[top]] = mat[i][arg[top]]
        return knnmat

    def __call__(self, data):

        X = data.x.cpu().detach().numpy().copy()

        S1 = cosine_similarity(X)
        S1 = self.makeKnn(S1, 80)
        self.normalized(S1)

        S2 = utils.to_dense_adj(data.edge_index)[0]
        S2 = S2.cpu().detach().numpy().copy()
        self.normalized(S2)

        S = (S1+S2)/2.
        S = (S+S.T)/2.

        D = np.diag(np.sum(S,axis=0))
        Ls = D-S
        Ls_norm = np.dot(np.sqrt(linalg.inv(D)),np.dot(Ls,np.sqrt(linalg.inv(D))))

        eigen_val, eigen_vec = eigsh(Ls_norm, self.num_clusters, which="SM")
        k_means = KMeans(n_clusters=self.num_clusters, n_init=10, tol=0.0000001)
        k_means.fit(eigen_vec)
        data.pseudo_label = k_means.labels_

        label = data.y.cuda().cpu().detach().numpy().copy()
        nmi = clus.normalized_mutual_info_score(label, k_means.labels_)
        print('\tDone process : make pseudo label (nmi: {:.3f})'.format(nmi))

        return data

class ExtractAttribute:
    def __init__(self, num_clusters, tree_depth):
        self.num_clusters = num_clusters
        self.tree_depth = tree_depth

    def __call__(self, data):

        # if dataset.x is tf-idf, not bag of words, we transform it to bag of words
        data.x = torch.where(data.x > 0., 1., 0.)

        if(self.tree_depth==-1):
            return data

        clf = DecisionTreeClassifier(max_depth=self.tree_depth)
        clf = clf.fit(data.x, data.pseudo_label)
        f_importance = clf.feature_importances_
        hit_idxes = [idx for idx, val in enumerate(
            f_importance) if val > 0]

        data.x = data.x[:, hit_idxes]

        print('\tDone process : extract attribute')
        return data


class MaskGraph:
    def __init__(self, mask_rate_node, mask_rate_edge):
        self.mask_rate_node = mask_rate_node
        self.mask_rate_edge = mask_rate_edge

    def __call__(self, data):

        num_nodes = data.x.size()[0]
        data.masked_x = data.x.clone()
        data.masked_edge_index = data.edge_index.clone()

        # sample some distinct nodes to be masked
        if(self.mask_rate_node > 0.):
            n_attribute = data.masked_x.size()[1]
            sample_size = int(num_nodes * self.mask_rate_node)
            masked_node_idxes = random.sample(range(num_nodes), sample_size)

            masked_node_labels_list = []
            for idx in masked_node_idxes:
                masked_node_labels_list.append(
                    data.masked_x[idx].clone().view(1, -1))
                data.masked_x[idx] = torch.zeros(n_attribute, dtype=torch.float)
            data.masked_node_label = torch.cat(masked_node_labels_list, dim=0)
            data.masked_node_idxes = torch.tensor(masked_node_idxes)

        # sample some distinct edges to be masked, based on mask rate
        if(self.mask_rate_edge > 0.):
            num_edges = int(data.masked_edge_index.size()[1])
            sample_size = int(num_edges * self.mask_rate_edge)
            ratio_positive_negative = 1

            pair_list = set(itertools.combinations(range(num_nodes), 2))
            edge_pair_list = set([(u, v)
                                  for u, v in data.masked_edge_index.numpy().T])
            noedge_pair_list = pair_list - edge_pair_list

            masked1 = random.sample(edge_pair_list, sample_size)
            masked2 = random.sample(noedge_pair_list, sample_size*ratio_positive_negative)
            masked_edge_idxes = list(masked1) + list(masked2)
            data.mask_edge_idxes = torch.tensor(
                [[u, v] for u, v in masked_edge_idxes])
            data.mask_edge_label = torch.cat([torch.ones(sample_size, dtype=torch.float),
                                        torch.zeros(sample_size*ratio_positive_negative, 
                                        dtype=torch.float)], dim=0)

            A = utils.to_dense_adj(data.masked_edge_index)[0]
            for (u, v) in masked1:
                A[u][v] = 0
                A[v][u] = 0
            data.masked_edge_index = utils.dense_to_sparse(A)[0]
        
        print('\tDone process : mask graph')
        return data


class ExtractSubstructureContextPair:
    def __init__(self, c, l1, device):
        self.c = c
        self.l1 = l1
        self.device = device

    def __call__(self, data):

        # extract useful attributes for clustering (hit idx) of original attribute
        pseudo_label = spectral_clustering(data, self.c)
        data = extract_attribute(data, pseudo_label, 6)

        # augument original graph based on BA and TF algorithm
        data = augument_graph(data, 2300, 1, 10)

        G = graph_data_obj_to_nx(data)

        data.list = []  # i-th object of this list means a graph data of i-th cluster
        for c_i in range(self.c):
            data_c_i = Data()

            idxes_of_c_i = np.where(pseudo_label == c_i)[0]
            G_c_i = G.subgraph(idxes_of_c_i)

            # select center_node of G_c_i based on Pagerank
            nodes_rank = nx.pagerank_scipy(G_c_i, alpha=0.85)
            center_node_idx = max((v, k) for k, v in nodes_rank.items())[1]
            data_c_i.center_idx = center_node_idx

            # Get context that is between l1 and the max diameter of the G_c_i
            l1_node_idxes = nx.single_source_shortest_path_length(G_c_i, center_node_idx,
                                                                  self.l1).keys()
            l2_node_idxes = idxes_of_c_i
            context_node_idxes = set(l1_node_idxes).symmetric_difference(
                set(l2_node_idxes))
            context_G_c_i = G_c_i.subgraph(context_node_idxes)
            # plot_G_contextG_pair(G_c_i, context_G_c_i, center_node_idx, c_i)
            context_G_c_i, context_node_map_c_i = reset_idxes(
                context_G_c_i)
            context_data = nx_to_graph_data_obj(context_G_c_i)
            data_c_i.x_context = context_data.x.to(self.device)
            data_c_i.edge_index_context = context_data.edge_index.to(
                self.device)

            # Get indices of overlapping nodes between G_c_i and context_G_c_i,
            context_substruct_overlap_idxes = list(context_node_idxes)
            context_substruct_overlap_idxes_reorder = [context_node_map_c_i[old_idx]
                                                       for old_idx in context_substruct_overlap_idxes]
            data_c_i.context_idxes = torch.tensor(
                context_substruct_overlap_idxes_reorder)

            data.list.append(data_c_i)

        return data


def nx_to_graph_data_obj(g):

    # nodes
    n_nodes = g.number_of_nodes()
    n_attributes = len(g.nodes[0])
    x = np.zeros((n_nodes, n_attributes))
    for i in range(n_nodes):
        x[i] = np.array([attribute for attribute in g.nodes[i].values()])
    x = torch.tensor(x, dtype=torch.float)

    # edges
    edges_list = []
    for i, j in g.edges():
        edges_list.append((i, j))
        edges_list.append((j, i))
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    # construct data obj
    data = Data(x=x, edge_index=edge_index)

    return data


def graph_data_obj_to_nx(data):

    G = nx.Graph()

    x = data.x.cpu().numpy()
    for idx, attributes in enumerate(x):
        G.add_node(idx)
        attr = {idx: {dim_idx: attribute for dim_idx,
                      attribute in enumerate(attributes)}}
        nx.set_node_attributes(G, attr)

    edge_index = data.edge_index.cpu().numpy()
    n_edges = edge_index.shape[1]
    for j in range(0, n_edges):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx)

    return G


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


data_path = 'D:\python\GCN\DeepGraphClustering\data'


def remake_to_labelorder(pred_tensor: torch.tensor, label_tensor: torch.tensor) -> dict:
    pred_ = pred_tensor.cuda().cpu().detach().numpy().copy()
    pred_ = np.array([np.argmax(i) for i in pred_])
    label_ = label_tensor.cuda().cpu().detach().numpy().copy()
    n_of_clusters = max(label_)+1
    pred_ids, label_ids = {}, {}
    for vid, (pred_id, label_id) in enumerate(zip(pred_, label_)):
        if(pred_id in pred_ids):
            pred_ids[pred_id].append(vid)
        else:
            pred_ids[pred_id] = []
            pred_ids[pred_id].append(vid)
        if(label_id in label_ids):
            label_ids[label_id].append(vid)
        else:
            label_ids[label_id] = []
            label_ids[label_id].append(vid)

    pred_pairs, label_pairs = [set() for _ in range(n_of_clusters)], [
        set() for _ in range(n_of_clusters)]
    for pred_key, label_key in zip(pred_ids.keys(), label_ids.keys()):
        pred_pairs[pred_key] |= set(
            [pair for pair in itertools.combinations(pred_ids[pred_key], 2)])
        label_pairs[label_key] |= set(
            [pair for pair in itertools.combinations(label_ids[label_key], 2)])

    table = np.array([[len(label_pair & pred_pair)
                       for label_pair in label_pairs] for pred_pair in pred_pairs])

    G = nx.DiGraph()
    G.add_node('s', demand=-n_of_clusters)
    G.add_node('t', demand=n_of_clusters)
    for pred_id in range(n_of_clusters):
        G.add_edge('s', 'p_{}'.format(pred_id), weight=0, capacity=1)
    for source, weights in enumerate(table):
        for target, w in enumerate(weights):
            G.add_edge('p_{}'.format(source), 'l_{}'.format(
                target), weight=-w, capacity=1)
    for label_id in range(n_of_clusters):
        G.add_edge('l_{}'.format(label_id), 't', weight=0, capacity=1)

    clus_label_map = {}
    result = nx.min_cost_flow(G)
    for i, d in result.items():
        for j, f in d.items():
            if f and i[0] == 'p' and j[0] == 'l':
                clus_label_map[int(i[2])] = int(j[2])
    w = torch.FloatTensor([[1 if j == clus_label_map[i] else 0 for j in range(n_of_clusters)]
                           for i in range(n_of_clusters)]).cuda()
    return torch.mm(pred_tensor, w)


def fuzzy_cmeans(data, n_of_clusters, *, m=1.07):
    n_of_clusters = n_of_clusters.cuda().cpu().detach().numpy().copy()
    result = cmeans(data.T, n_of_clusters, m, 0.001, 10000, seed=0)
    fuzzy_means = result[1].T
    result = torch.FloatTensor(np.array(fuzzy_means)).cuda()
    return result

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

def clustering_accuracy(labels, preds, num_class):
    def make_tree(set_, map_, map_list):
        if(len(set_) > 0):
            for i in set_:
                map_.append(i)
                set__ = copy.copy(set_)
                set__.remove(i)
                make_tree(set__, map_, map_list)
                map_.remove(i)
        else:
            map_list.append(map_.copy())

    label_set = [i for i in range(num_class)]
    map_list = []
    make_tree(label_set, [], map_list)
    
    acc_max = 0.
    for map_ in map_list:
        map_ = {cluster_id : class_id for cluster_id, class_id in enumerate(map_)}
        pred_mapped = list(map(map_.get, preds))
        acc = accuracy_score(pred_mapped, labels)
        if(acc > acc_max):
            acc_max = acc
    
    return acc_max


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
