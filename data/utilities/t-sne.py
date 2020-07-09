import argparse
import scipy as sp
from sklearn.datasets import fetch_mldata
import sklearn.base
import bhtsne
import matplotlib.pyplot as plot
import numpy as np

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='virtualize samples from data by using t-sne model')
    parser.add_argument('data', type=str, help='<<< dataset_name(~.csv) in directory data/experiments/')
    parser.add_argument('label', type=str, help='<<< label_name(~.csv) in directory data/experiments/')
    parser.add_argument('-f', '--filename', type=str, default='virtualize.png', help='<<< file name(~.png) when plt.savefig')
    args = parser.parse_args()

    data = np.loadtxt('../experiment/' + args.data)
    label = np.loadtxt('../experiment/' + args.label)
    
    bh = BHTSNE(dimensions=2, perplexity=30.0, theta=0.5, rand_seed=-1, max_iter=10000)
    tsne = bh.fit_transform(data)

    xmin = tsne[:,0].min()
    xmax = tsne[:,0].max()
    ymin = tsne[:,1].min()
    ymax = tsne[:,1].max()

    plot.figure( figsize=(16,12) )
    plot.scatter(tsne[:,0], tsne[:,1], c=label)
    plot.axis([xmin,xmax,ymin,ymax])
    plot.xlabel("component 0")
    plot.ylabel("component 1")
    plot.title("data visualization")
    plot.savefig("../experiment/" + args.filename)
