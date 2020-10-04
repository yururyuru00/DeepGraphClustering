import matplotlib.pyplot as plt
import networkx as nx
import sklearn.base
import bhtsne
import scipy as sp
import numpy as np
from matplotlib.colors import ListedColormap


def plot_G_contextG_pair(G, context_G, center_substract_idx, c_i):

    nodes_context_G = [n_i for n_i in context_G.nodes]
    color_map = []
    for node in G:
        if node == center_substract_idx:
            color_map.append('red')
        elif node in nodes_context_G:
            color_map.append('blue')
        else:  # when node is in a substract graph G
            color_map.append('orange')

    plt.subplot(111)
    nx.draw(G, node_color=color_map,
            with_labels=True)
    plt.savefig('./data/experiment/test/G_context_G_pair_c{}.png'.format(c_i))


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=100):
        self.dimensions = dimensions
        self.perplexity = perplexity
        self.theta = theta
        self.rand_seed = rand_seed
        self.max_iter = max_iter
        print(self.max_iter)

    def fit_transform(self, X):
        return bhtsne.tsne(
            X.astype(sp.float64),
            dimensions=self.dimensions,
            perplexity=self.perplexity,
            theta=self.theta,
            rand_seed=self.rand_seed
        )


def plot_Zn(data, label, path_save='./test'):
    bh = BHTSNE(dimensions=2, perplexity=30.0,
                theta=0.5, rand_seed=-1, max_iter=10000)
    data = bh.fit_transform(data)

    xmin = data[:, 0].min()
    xmax = data[:, 0].max()
    ymin = data[:, 1].min()
    ymax = data[:, 1].max()

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    cmap = ListedColormap(colors)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data[:, 0], data[:, 1], cmap=cmap, c=label)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_xlabel("component 0")
    ax.set_ylabel("component 1")
    ax.set_title("data visualization")
    ax.legend(loc='upper left')
    plt.savefig(path_save + '.png')
