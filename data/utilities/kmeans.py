import argparse
import numpy as np
from sklearn.cluster import KMeans

def purity(preds, labels):
    size = len(preds)
    clus_size = np.max(preds)+1
    clas_size = np.max(labels)+1
    
    table = np.zeros((clus_size, clas_size))
    for i in range(size):
        table[preds[i]][labels[i]] += 1
    sum = 0
    for k in range(clus_size):
        sum += np.amax(table[k])
    return sum/size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='clustering dataset by using k-means')
    parser.add_argument('data', type=str, help='<<< dataset_name(~.csv) in directory data/experiments/')
    parser.add_argument('clusters', type=int, help='<<< number of clusters what you want')
    parser.add_argument('label', type=str, help='<<< label_name(~.csv) in directory data/experiments/')
    args = parser.parse_args()

    X = np.loadtxt('../experiment/' + args.data)
    labels = np.loadtxt('../experiment/' + args.label, dtype=np.int32)

    pred = KMeans(n_clusters=args.clusters).fit_predict(X)
    print(purity(pred, labels))