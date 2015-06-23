#from sklearn import cluster, decomposition
#import pylab as pl

#HiC1 = [[8.0,8.0,15,90,22.4008378422],
#[8.0,8.0,20,60,25.2343594579],
#[10.0,8.0,25,60,25.6829557058],
#[10.0,10.0,15,30,26.3230279732],
#[8.0,10.0,20,90,26.8309739422],
#[8.0,10.0,15,10,27.7811668691],
#[10.0,8.0,15,10,28.4787107887],
#[8.0,10.0,15,60,28.8044597892],
#[8.0,8.0,25,90,28.9396691924],
#[8.0,8.0,25,30,28.9695153018],
#[8.0,10.0,20,30,28.9917787393],
#[8.0,10.0,25,90,29.3386636405],
#[8.0,10.0,20,60,29.5693532596],
#[8.0,10.0,25,10,30.0181270455],
#[8.0,8.0,20,30,30.2030315636],
#[10.0,10.0,20,30,30.5495686888],
#[8.0,10.0,20,10,30.6853334208],
#[10.0,8.0,20,30,30.8367759548],
#[8.0,8.0,20,10,30.8879572909],
#[10.0,8.0,20,60,30.9217899424],
#[10.0,10.0,25,60,30.9474424318],
#[10.0,10.0,15,90,31.3581787391],
#[8.0,8.0,20,90,31.5419781773],
#[10.0,10.0,25,30,31.6490087561],
#[10.0,8.0,15,60,31.7147473609],
#[10.0,8.0,25,90,32.3917624029],
#[10.0,8.0,15,90,32.659017816],
#[10.0,8.0,15,30,32.8371656214],
#[8.0,8.0,15,60,33.1244535603],
#[8.0,8.0,15,30,33.3494159934],
#[8.0,8.0,15,10,33.3761112299],
#[10.0,10.0,20,90,33.7790862698],
#[10.0,10.0,20,60,33.9753786497],
#[10.0,10.0,25,10,34.0490181734],
#[8.0,8.0,25,60,34.8702516154],
#[10.0,8.0,25,10,35.0451389491],
#[8.0,10.0,15,30,35.0820375393],
#[10.0,8.0,20,90,35.4618943388],
#[10.0,10.0,15,60,35.5669791409],
#[10.0,8.0,25,30,35.6169425313],
#[8.0,8.0,25,10,35.8043555547],
#[10.0,10.0,20,10,35.8099566759],
#[8.0,10.0,25,60,36.2083388313],
#[8.0,10.0,25,30,37.0995008261],
#[10.0,10.0,15,10,37.3915136467],
#[10.0,8.0,20,10,39.8884178163],
#[8.0,10.0,15,90,41.7073641969]]

#k_means = cluster.KMeans(k=5)
#k_means.fit(HiC1) 
#for i in range(len(HiC1)): print HiC1[i], k_means.labels_[i]

#pca = decomposition.PCA(n_components=3)
#pca.fit(HiC1)
#X = pca.transform(HiC1)

#pl.scatter(X[:, 0], X[:, 1], c = X[0]) 
#pl.show()




"""
====================================================================
Comparison of the K-Means and MiniBatchKMeans clustering algorithms
"""

import time

import numpy as np
import pylab as pl

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs

##############################################################################
## Generate sample data
np.random.seed(0)

batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)

##############################################################################

HiC1 = [[8.0,8.0,15,90,22.4008378422],
[8.0,8.0,20,60,25.2343594579],
[10.0,8.0,25,60,25.6829557058],
[10.0,10.0,15,30,26.3230279732],
[8.0,10.0,20,90,26.8309739422],
[8.0,10.0,15,10,27.7811668691],
[10.0,8.0,15,10,28.4787107887],
[8.0,10.0,15,60,28.8044597892],
[8.0,8.0,25,90,28.9396691924],
[8.0,8.0,25,30,28.9695153018],
[8.0,10.0,20,30,28.9917787393],
[8.0,10.0,25,90,29.3386636405],
[8.0,10.0,20,60,29.5693532596],
[8.0,10.0,25,10,30.0181270455],
[8.0,8.0,20,30,30.2030315636],
[10.0,10.0,20,30,30.5495686888],
[8.0,10.0,20,10,30.6853334208],
[10.0,8.0,20,30,30.8367759548],
[8.0,8.0,20,10,30.8879572909],
[10.0,8.0,20,60,30.9217899424],
[10.0,10.0,25,60,30.9474424318],
[10.0,10.0,15,90,31.3581787391],
[8.0,8.0,20,90,31.5419781773],
[10.0,10.0,25,30,31.6490087561],
[10.0,8.0,15,60,31.7147473609],
[10.0,8.0,25,90,32.3917624029],
[10.0,8.0,15,90,32.659017816],
[10.0,8.0,15,30,32.8371656214],
[8.0,8.0,15,60,33.1244535603],
[8.0,8.0,15,30,33.3494159934],
[8.0,8.0,15,10,33.3761112299],
[10.0,10.0,20,90,33.7790862698],
[10.0,10.0,20,60,33.9753786497],
[10.0,10.0,25,10,34.0490181734],
[8.0,8.0,25,60,34.8702516154],
[10.0,8.0,25,10,35.0451389491],
[8.0,10.0,15,30,35.0820375393],
[10.0,8.0,20,90,35.4618943388],
[10.0,10.0,15,60,35.5669791409],
[10.0,8.0,25,30,35.6169425313],
[8.0,8.0,25,10,35.8043555547],
[10.0,10.0,20,10,35.8099566759],
[8.0,10.0,25,60,36.2083388313],
[8.0,10.0,25,30,37.0995008261],
[10.0,10.0,15,10,37.3915136467],
[10.0,8.0,20,10,39.8884178163],
[8.0,10.0,15,90,41.7073641969]]

# Compute clustering with Means

#k_means = KMeans(5) #init='k-means++', n_clusters=3, n_init=10)
k_means = KMeans(5,init='k-means++', n_init=10)
t0 = time.time()
k_means.fit(HiC1)
t_batch = time.time() - t0
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)

##############################################################################
# Compute clustering with MiniBatchKMeans

mbk = MiniBatchKMeans(5, init='k-means++', batch_size=batch_size, n_init=10, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(HiC1)
t_mini_batch = time.time() - t0
mbk_means_labels = mbk.labels_
mbk_means_cluster_centers = mbk.cluster_centers_
mbk_means_labels_unique = np.unique(mbk_means_labels)

##############################################################################
# Plot result

fig = pl.figure(figsize=(5, 3))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']

# We want to have the same colors for the same cluster from the
# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
# closest one.

distance = euclidean_distances(k_means_cluster_centers,
                               mbk_means_cluster_centers,
                               squared=True)
order = distance.argmin(axis=1)

# KMeans
ax = fig.add_subplot(1, 3, 1)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.', markersize = 12)
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=8)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
#pl.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
    #t_batch, k_means.inertia_))

# MiniBatchKMeans
ax = fig.add_subplot(1, 3, 2)
for k, col in zip(range(n_clusters), colors):
    my_members = mbk_means_labels == order[k]
    cluster_center = mbk_means_cluster_centers[order[k]]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.',markersize=12)
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=8)
ax.set_title('MiniBatchKMeans')
ax.set_xticks(())
ax.set_yticks(())
#pl.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
        #(t_mini_batch, mbk.inertia_))

# Initialise the different array to all False
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1, 3, 3)

for l in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_means_labels == order[k]))

identic = np.logical_not(different)
ax.plot(X[identic, 0], X[identic, 1], 'w',
        markerfacecolor='#bbbbbb', marker='.', markersize = 12)
ax.plot(X[different, 0], X[different, 1], 'w',
        markerfacecolor='m', marker='.', markersize = 8)
ax.set_title('Difference')
ax.set_xticks(())
ax.set_yticks(())

pl.show()