import logging

import tensorflow as tf


class Kmeans:
    EPSILON = 1e-07

    def __init__(self, k):
        self.centroids = None
        self.k = k

    @staticmethod
    def __get_clusters(data, centroids):
        clusters = []
        for sample in data:
            distances = tf.norm(tf.expand_dims(sample, axis=0) - centroids, ord="euclidean", axis=1)

            clusters.append(tf.argmin(distances, axis=0))

        return tf.stack(clusters)

    @staticmethod
    def __initialize_centroids(data, k):
        """
        returns list of centroid coordinates. They are randomly selected from the data.
        :param data: the actual dataset to cluster
        :param k: number of clusters
        :return: clusters (their coordinates)
        """
        centroid_indices = tf.random.uniform([k], 0, data.shape[0], dtype=tf.int32)
        return tf.gather(data, centroid_indices)

    @staticmethod
    def __update_centroids(data, clusters, centroids):
        """
        Updates the cluster centroids
        :param data:
        :param clusters:
        :param centroids:
        :return:
        """
        new_centroids = []
        centroid_distance = []
        for i in range(centroids.shape[0]):
            cluster_data = data[tf.equal(i, clusters)]
            cluster_centroid = tf.reduce_mean(cluster_data, axis=0)
            new_centroids.append(cluster_centroid)
            centroid_dist = tf.norm(cluster_centroid - centroids[i], ord="euclidean")
            centroid_distance.append(centroid_dist)
        return tf.stack(new_centroids), tf.reduce_sum(tf.squeeze(centroid_distance))

    def __reset_empty_centroids(self, non_empty_clusters, data):
        empty_clusters = set(range(self.k)) - set(non_empty_clusters.numpy())
        new_centroids_indices = tf.random.uniform([len(empty_clusters)], 0, data.shape[0], dtype=tf.int32)
        new_centroids = self.centroids.numpy()
        idx = 0
        for i in empty_clusters:
            new_centroids[i] = tf.gather(data, tf.gather(new_centroids_indices, idx))
            idx += 1
        self.centroids = tf.stack(new_centroids)

    def fit_transform(self, data, max_iter=200):
        if self.centroids is None:
            self.centroids = Kmeans.__initialize_centroids(data, self.k)
        clusters = None
        for i in range(max_iter):
            clusters = Kmeans.__get_clusters(data, self.centroids)
            y, _ = tf.unique(clusters)
            while y.shape[0] < self.k:
                logging.debug("resetting centroids at iter {}".format(i))
                self.__reset_empty_centroids(y, data)
                clusters = Kmeans.__get_clusters(data, self.centroids)
                y, _ = tf.unique(clusters)
                
            self.centroids, centroid_evolution = Kmeans.__update_centroids(data, clusters, self.centroids)
            logging.debug("Centroid evo: {}".format(centroid_evolution))
            if centroid_evolution <= Kmeans.EPSILON:
                break

        return clusters
