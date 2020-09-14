import argparse
import scipy as sp
from sklearn.datasets import fetch_mldata
import sklearn.base
import bhtsne
import matplotlib.pyplot as plt
import numpy as np
import itertools
import networkx as nx
import re

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

def t_sne(data, *, dim=2):
    bh = BHTSNE(dimensions=dim, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=10000)
    tsne = bh.fit_transform(data)
    
    return tsne

def plot_feature(data, label, savename, mode):
    
    def target_to_color(target):
        if type(target) == np.ndarray:
            return (target[0], target[1], target[2])
        else:
            return "rgb"[target]

    x = t_sne(data)
    xmin = x[:,0].min()
    xmax = x[:,0].max()
    ymin = x[:,1].min()
    ymax = x[:,1].max()

    c_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    ary_x = [[] for _ in range(np.max(label)+1)]
    for x_i, l in zip(x, label):
        ary_x[l].append(x_i)
    ary_x = [np.array([[x_i[0], x_i[1]] for x_i in x]) for x in ary_x]

    fig = plt.figure( figsize=(16,12) )
    ax = fig.add_subplot(1,1,1)
    if(mode=='hard'):
        for l_id, samples in enumerate(ary_x):
            ax.scatter(samples[:, 0], samples[:, 1], c=c_map[l_id], label='c{}'.format(l_id))
    else:
        ax.scatter(x[:, 0], x[:, 1], c=[target_to_color(c) for c in label])
    ax.axis([xmin,xmax,ymin,ymax])
    ax.set_xlabel("component 0")
    ax.set_ylabel("component 1")
    ax.set_title("data visualization")
    ax.legend(loc='upper left')
    plt.savefig(savename + '.png')

def load_path(data_path, label_path, savename, mode):
    
    def load(data_path, type):
        if(type == 'csv'): return np.loadtxt(data_path)
        else: return np.load(data_path)
    
    data_type = re.findall(r'.+\.([a-z]{3})', data_path)[0]
    label_type = re.findall(r'.+\.([a-z]{3})', label_path)[0]
    data = load(data_path, data_type)
    #label = [l for l in map(int, load(label_path, label_type))]
    label = [np.argmax(pred) for pred in np.load(label_path)]

    plot_feature(data, label, savename, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, 
                                    help='data_path ubtil data')
    parser.add_argument('label_path', type=str, 
                                    help='label_path ubtil label')
    parser.add_argument('--save', type=str, default='test',
                                    help='filename when save')
    parser.add_argument('--mode', type=str, default='hard',
                                    help='hard or soft clustering visualization')

    args = parser.parse_args()
    load_path(args.data_path, args.label_path, args.save, args.mode)