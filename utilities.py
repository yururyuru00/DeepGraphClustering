import random
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from skfuzzy.cluster import cmeans
import sklearn.metrics.cluster as clus
import itertools
import networkx as nx
from torch_geometric.data import Data
from torch_geometric import utils
from scipy.sparse.linalg import eigsh


from debug import plot_G_contextG_pair


class Mask:
    def __init__(self, mask_rate_node, mask_rate_edge):
        self.mask_rate_node = mask_rate_node
        self.mask_rate_edge = mask_rate_edge

    def __call__(self, data):

        num_nodes = data.x.size()[0]

        # sample some distinct nodes to be masked, based on mask rate
        if(self.mask_rate_node > 0.):
            sample_size = int(num_nodes * self.mask_rate_node)
            masked_node_idxes = random.sample(range(num_nodes), sample_size)

            # get pseudo label
            pseudo_label = SpectralClustering(data)

            clf = DecisionTreeClassifier(max_depth=8)
            clf = clf.fit(data.x, pseudo_label)
            f_importance = clf.feature_importances_
            hit_idxes = [idx for idx, val in enumerate(
                f_importance) if val > 0]

            data.x = data.x[:, hit_idxes]

            mask_node_labels_list = []
            for idx in masked_node_idxes:
                mask_node_labels_list.append(
                    data.x[idx].view(1, -1))
            data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
            data.masked_node_idxes = torch.tensor(masked_node_idxes)

            # mask the original node feature of the masked node
            n_attribute = data.x.size()[1]
            for idx in masked_node_idxes:
                data.x[idx] = torch.zeros(len(hit_idxes), dtype=torch.float)

        # sample some distinct edges to be masked, based on mask rate
        if(self.mask_rate_edge > 0.):
            num_edges = int(data.edge_index.size()[1])
            sample_size = int(num_edges * self.mask_rate_edge)

            pair_list = set(itertools.combinations(range(num_nodes), 2))
            edge_pair_list = set([(u, v)
                                  for u, v in data.edge_index.numpy().T])
            noedge_pair_list = pair_list - edge_pair_list

            masked1 = random.sample(edge_pair_list, sample_size)
            masked2 = random.sample(noedge_pair_list, sample_size*5)
            masked_edge_idxes = list(masked1) + list(masked2)
            data.mask_edge_idxes = torch.tensor(
                [[u, v] for u, v in masked_edge_idxes])
            data.mask_edge_label = torch.cat([torch.ones(sample_size, dtype=torch.float),
                                              torch.zeros(sample_size*5, dtype=torch.float)], dim=0)

            A = utils.to_dense_adj(data.edge_index)[0]
            for (u, v) in masked1:
                A[u][v] = 0
                A[v][u] = 0
            data.edge_index = utils.dense_to_sparse(A)[0]

        return data


class ExtractSubstructureContextPair:
    def __init__(self, c, l1, device):
        self.c = c
        self.l1 = l1
        self.device = device

    def __call__(self, data):
        print('transformer called')
        G = graph_data_obj_to_nx(data)

        # make data and graph object for each i-th cluster (c_i)
        '''data_np = data.x.cuda().cpu().detach().numpy().copy()
        k_means = KMeans(self.c, n_init=10, random_state=0, tol=0.0000001)
        k_means.fit(data_np)
        cluster_label = k_means.labels_'''
        cluster_label = data.y

        data.list = []  # i-th object of this list means a graph data of i-th cluster
        for c_i in range(self.c):
            data_c_i = Data()

            idxes_of_c_i = np.where(cluster_label == c_i)[0]
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
            plot_G_contextG_pair(G_c_i, context_G_c_i, center_node_idx, c_i)
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
    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
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


def normalized(mat):
    for i in range(len(mat)):
        sum_ = np.sum(mat[i])
        for j in range(len(mat[i])):
            if(sum_ == 0):
                mat[i][j] = 0.
            else:
                mat[i][j] = mat[i][j] / sum_


def makeLaplacian(mat):
    L = np.zeros((len(mat), len(mat[0])))
    for d in range(len(mat)):
        for i in range(len(mat)):
            L[d][d] += (mat[i][d] + mat[d][i])
        L[d][d] = L[d][d]/2.0
    for i in range(len(L)):
        for j in range(len(L[0])):
            L[i][j] -= (mat[i][j]+mat[j][i])/2.
    return L


def makeNormLaplacian(mat):
    L = makeLaplacian(mat)
    d = np.zeros(len(mat))
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            d[i] += (mat[i][j] + mat[j][i])
        d[i] = d[i] / 2.
    for i in range(len(L)):
        for j in range(len(L[i])):
            if(d[i] == 0 or d[j] == 0):
                L[i][j] = 0.
            else:
                L[i][j] = (1./np.sqrt(d[i])) * L[i][j] * (1./np.sqrt(d[j]))
    return L


def makeKnn(mat, k):
    knnmat = np.zeros((len(mat), len(mat[0])))
    for i in range(len(mat)):
        arg = np.argsort(-mat[i])
        for top in range(k):
            knnmat[i][arg[top]] = mat[i][arg[top]]
    return knnmat


def SpectralClustering(data):
    obj_size = data.x.size()[0]
    features = data.x.cpu().detach().numpy().copy()

    S1 = np.zeros((obj_size, obj_size))
    for i in range(obj_size):
        for j in range(i+1, obj_size):
            S1[i][j] = np.dot(features[i], features[j]) / \
                (np.linalg.norm(features[i]) * np.linalg.norm(features[j]))
            S1[j][i] = S1[i][j]
    for d in range(len(S1)):
        S1[d][d] = 0.

    S1 = makeKnn(S1, 80)
    normalized(S1)

    S2 = utils.to_dense_adj(data.edge_index)[0]
    S2 = S2.cpu().detach().numpy().copy()
    normalized(S2)

    S = (S1+S2)/2.

    Ls = makeNormLaplacian(S)
    eigen_val, eigen_vec = eigsh(Ls, 7, which="SM")

    k_means = KMeans(n_clusters=7, n_init=10, tol=0.0000001)
    k_means.fit(eigen_vec)

    return k_means.labels_


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
    adj = normalize(adj + sp.eye(adj.shape[0]))  # ここでA = A+I 更に D^-1*A までしてる

    # sampling 3 clusters for fuzzy means visualization
    labels = np.where(labels)[1]
    features, adj = features.toarray(), adj.toarray()
    idx, del_idx = [], []
    for v_id, label_id in enumerate(labels):
        if(label_id in [0, 2, 5]):
            idx.append(v_id)
        else:
            del_idx.append(v_id)
    dane_emb = np.loadtxt('./data/experiment/DANEemb.csv')
    dane_emb = np.delete(dane_emb, del_idx, axis=0)
    features = np.delete(features, del_idx, axis=0)
    adj = np.delete(adj, del_idx, axis=0)
    adj = np.delete(adj, del_idx, axis=1)
    labels = np.delete(labels, del_idx, axis=0)
    map_ = {0: 0, 2: 1, 5: 2}
    labels = [relabeled for relabeled in map(lambda x:map_[x], labels)]

    dane_emb = torch.FloatTensor(dane_emb)
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    adj = torch.FloatTensor(adj)  # ここで各入力A, X, lをtensor型に変更

    return adj, features, labels, dane_emb


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
