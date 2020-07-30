from fcmeans import FCM
from skfuzzy.cluster import cmeans
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
import numpy as np
import argparse
from data_plot import t_sne

def target_to_color(target):
    if type(target) == np.ndarray:
        return (target[0], target[1], target[2])
    else:
        return "rgb"[target]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, 
                                    help='data_path ubtil data')
    parser.add_argument('n_clusters', type=int, 
                                    help='number of clusters')                                
    parser.add_argument('--save', type=str, default='test',
                                    help='filename when save')
    parser.add_argument('--m', type=float, default='1.5',
                                    help='fuzzy coefficient (hyparparameter)')
    args = parser.parse_args()
    X = np.load(args.data_path)

    # fit the fuzzy-c-means
    '''fcm = FCM(n_clusters=args.n_clusters)
    fcm.fit(X)
    label = [np.argmax(label) for label in fcm.u]'''
    cm_result = cmeans(X.T, args.n_clusters, args.m, 0.001, 10000)
    soft_cluster = cm_result[1].T
    hard_cluster = [np.argmax(c) for c in soft_cluster]
    '''k_means = KMeans(args.n_clusters, n_init=10, random_state=0, tol=0.0000001)
    k_means.fit(X)
    label = k_means.labels_'''

    # make lower dimension by t-sne for visualization
    X = t_sne(X)

    # plot result
    fig = plt.figure(figsize=(16,12))
    plt.scatter(X[:,0], X[:,1], c=[target_to_color(c) for c in soft_cluster])
    plt.savefig(args.save + '.png')