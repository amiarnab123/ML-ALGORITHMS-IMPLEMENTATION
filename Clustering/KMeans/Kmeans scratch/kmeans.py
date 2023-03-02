import random
import numpy as np

class Kmeans:

    def __init__(self,n_cluster = 2,max_iter = 100):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self,X):
        random_index = random.sample(range(0,X.shape[0]),self.n_cluster)
        self.centroids = X[random_index]
        
        ## assign cluster
        for i in range(self.max_iter):
            cluster_group = self.Assign_cluster(X)

            ## move centroids
            old_centroid = self.centroids 
            self.centroids = self.move_centroids(X,cluster_group)

            ## cheak finish
            if (old_centroid == self.centroids).all():
                break

        return cluster_group

    def Assign_cluster(self,X):
        distances = []
        cluster_group = []

        for row in X:
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid,row-centroid)))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()
        
        return np.array(cluster_group)
    

    def move_centroids(self,X,cluster_group):
        new_centroids = []

        cluster_type = np.unique(cluster_group)
        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis = 0))

        return np.array(new_centroids)