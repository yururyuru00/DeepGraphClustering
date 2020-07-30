import argparse
import scipy as sp
from sklearn.datasets import fetch_mldata
import sklearn.base
import bhtsne
import matplotlib.pyplot as plt
import numpy as np
import itertools
import networkx as nx

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

def t_sne(data):
    bh = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=10000)
    tsne = bh.fit_transform(data)
    
    return tsne

def plot_feature(data_path, label_path, *, savename='./feature.png'):
    label =  np.load(label_path)
    data = np.load(data_path)
    x = t_sne(data)

    xmin = x[:,0].min()
    xmax = x[:,0].max()
    ymin = x[:,1].min()
    ymax = x[:,1].max()

    plt.figure( figsize=(16,12) )
    plt.scatter(x[:,0], x[:,1], c=label)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xlabel("component 0")
    plt.ylabel("component 1")
    plt.title("data visualization")
    plt.savefig(savename + '.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, 
                                    help='data_path ubtil data')
    parser.add_argument('label_path', type=str, 
                                    help='label_path ubtil label')
    parser.add_argument('--save', type=str, default='test',
                                    help='filename when save')

    args = parser.parse_args()
    plot_feature(args.data_path, args.label_path, savename=args.save)