# Creator: Atif Hassan
# email-id: atif.hassan@iitkgp.ac.in
# email-id: atif.hit.hassan@gmail.com

import numpy as np
import math
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

def icroust(X, Y, points_to_remove, n_neighbours, points_to_remove_at_a_time, num_clusters=10000, maj_class=0, min_class=1):
    X_minority = X[np.where(Y==min_class)[0]]
    X_majority = X[np.where(Y==maj_class)[0]]
    indices = list()
    if cluster_centers > len(X_minority):
        dists_cent = np.array([min([np.linalg.norm(i-j) for j in X_minority]) for i in X_majority])
    else:
        #Cluster minority samples
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, max_iter=100, random_state=0).fit(X_minority)
        #1. Find centroids
        centroids = kmeans.cluster_centers_
        #2. Get distance of majority class samples from closest centroid
        dists_cent = np.array([min([np.linalg.norm(i-j) for j in centroids]) for i in X_majority])

    print("Points to remove = "+str(points_to_remove))
    #for k in range(0, points_to_remove, points_to_remove_at_a_time):
    for k in range(math.floor(points_to_remove/points_to_remove_at_a_time)):
        #3. Sort majority samples by above distance in ascending order (could be important samples. Sort in descending order ?)
        #4. Get average distance with k=3 nearest neighbour of majority samples
        neigh = NearestNeighbors(n_neighbors=n_neighbours+1, algorithm='kd_tree', n_jobs=-1).fit(X_majority)
        dist, inds = neigh.kneighbors(X_majority)
        dists_nn = np.array([sum(dist[i])/n_neighbours for i in range(len(X_majority))])
        samples = [[i, dists_cent[i]+dists_nn[i]] for i in range(len(X_majority))]
        samples.sort(key=lambda z: z[1])
        #6. Now remove top k majority samples
        indices+= [i[0] for i in samples[:points_to_remove_at_a_time]]
        X_majority = np.delete(X_majority, [i[0] for i in samples[:points_to_remove_at_a_time]], axis=0)
    X, Y = list(), list()
    for i in X_majority:
        X.append(i)
        Y.append(maj_class)
    for i in X_minority:
        X.append(i)
        Y.append(min_class)
    return np.array(X), np.array(Y)
    

    return X, Y
