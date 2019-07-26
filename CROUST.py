# Creator: Atif Hassan
# email-id: atif.hassan@iitkgp.ac.in
# email-id: atif.hit.hassan@gmail.com

'''
Copyright [2019] [Atif Hassan]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

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
