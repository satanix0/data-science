import random
import numpy as np


class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        """
        Initialize KMeans clustering with specified number of clusters and maximum iterations.

        Parameters:
        - n_clusters: int, default=2. The number of clusters to form.
        - max_iter: int, default=100. The maximum number of iterations to run the algorithm.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        """
        Fit the model to data X and return cluster assignments for each data point.

        Parameters:
        - X: numpy array of shape (n_samples, n_features). Input data to cluster.

        Returns:
        - cluster_labels: numpy array of shape (n_samples,). Cluster labels for each data point.
        """
        # Step 1: Initialize centroids by randomly selecting 'n_clusters' points from the dataset
        random_indices = random.sample(range(0, X.shape[0]), self.n_clusters)
        # initializes the centroids with the row values at those indices
        self.centroids = X[random_indices]

        # Step 2: Iteratively update centroids and reassign points to clusters
        for i in range(self.max_iter):

            # Step 2.1: Assign each point to the nearest centroid
            cluster_groups = self.assign_clusters(X)

            # Step 2.2: Move centroids to the mean of points assigned to each cluster
            old_centroids = self.centroids
            self.centroids = self.move_centroids(X, cluster_groups)

            # Check Termination criterion (if centroids do not change)
            if (old_centroids == self.centroids).all():
                break
        # return the final clusters for each row
        return cluster_groups

    def assign_clusters(self, X):
        """
        Assign each data point to the nearest centroid.

        Parameters:
        - X: numpy array of shape (n_samples, n_features). Input data.

        Returns:
        - cluster_labels: numpy array of shape (n_samples,). Cluster labels for each data point.
        """

        cluster_groups = list()
        distances = []

        for row in X:
            for centroid in self.centroids:
                # Calculates Euclidean dist. between each row and each centroid (which is equual to n_clusters)
                distances.append(
                    np.sqrt(np.dot((row-centroid), (row-centroid))))
            # Calculates minimum distance out of n distances for eac row
            min_dist = min(distances)

            # getting the index which minimum dist is at (either 0 or 1)
            index_pos = distances.index(min_dist)
            cluster_groups.append(index_pos)

            # clearing distance after each row
            distances.clear()

        return np.array(cluster_groups)

    def move_centroids(self, X, cluster_groups):
        """
        Update centroid positions by calculating the mean of the points assigned to each cluster.

        Parameters:
        - X: numpy array of shape (n_samples, n_features). Input data.
        - cluster_labels: numpy array of shape (n_samples,). Cluster assignments.

        Returns:
        - new_centroids: numpy array of shape (n_clusters, n_features). Updated centroid positions.
        """
        # retuns unique values in the cluster_groups list i.e. all unique cluster labels
        cluster_type = np.unique(cluster_groups)

        # calculates column-wise mean of all the rows within each cluster i.e. if 100 rows, 45 are cluster 1, 55 cluster 2
        # then return mean of 45 rows and 55 rows over all columns.
        new_centroids = list()
        for type in cluster_type:
            new_centroids.append(X[cluster_groups == type].mean(axis=0))

        return np.array(new_centroids)
