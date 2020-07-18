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

def plot_feature(data, label, *, savename='feature.png'):
    bh = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=10000)
    tsne = bh.fit_transform(data)

    xmin = tsne[:,0].min()
    xmax = tsne[:,0].max()
    ymin = tsne[:,1].min()
    ymax = tsne[:,1].max()

    plt.figure( figsize=(16,12) )
    plt.scatter(tsne[:,0], tsne[:,1], c=label)
    plt.axis([xmin,xmax,ymin,ymax])
    plt.xlabel("component 0")
    plt.ylabel("component 1")
    plt.title("data visualization")
    plt.savefig("../experiment/" + savename + '.png')

def  plot_label(**kwargs):
    min__, max__ = 100, -1
    for pred in kwargs.values():
        min_, max_ = np.min(pred), np.max(pred)
        if(min_ < min__):
            min__ = min_
        if(max_ > max__):
            max__ = max_

    fig = plt.figure(figsize=(35, 17))
    ax = [fig.add_subplot(1, 3, 1),fig.add_subplot(1, 3, 2),fig.add_subplot(1, 3, 3)]
    for i, key in enumerate(kwargs.keys()):
        preds = kwargs[key][::100]
        for pred in preds:
            ax[i].plot([i for i in range(len(pred))], pred)
        ax[i].set_ylim(min__, max__)
    plt.savefig('./pred_label.png')

label = np.loadtxt('../experiment/DANElabel.csv')
for epoch in [0,1,2,49,50,51,52]:
    Zn = np.loadtxt('../experiment/epochs200_skips50/Zn_epoch#{}.csv'.format(epoch))
    plot_feature(Zn, label, savename='epochs200_skips50/Zn_epoch{}_label'.format(epoch))