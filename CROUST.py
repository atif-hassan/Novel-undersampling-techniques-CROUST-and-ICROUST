# Creator: Atif Hassan
# email-id: atif.hassan@iitkgp.ac.in
# email-id: atif.hit.hassan@gmail.com

from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import math

def croust(X, Y, points_to_remove, num_clusters=10000, maj_class=0, min_class=1):
    count_minority = len(X_minority)
    X_minority = X[np.where(Y==min_class)[0]]
    kmeans = MiniBatchKMeans(n_clusters=, max_iter=100, random_state=0)
    distances = kmeans.fit_transform(X)
    distances = [min(i) for i in distances]
    majority_samples = [[distances[i], i] for i in range(len(Y)) if Y[i] == maj_class]
    majority_samples = sorted(majority_samples, key=lambda x: x[0], reverse=True)
    
    to_remove_indices = [i[1] for i in majority_samples[:points_to_remove]]
    return np.delete(X, to_remove_indices, axis=0), np.delete(Y, to_remove_indices, axis=0)
