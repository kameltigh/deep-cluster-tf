import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA

from deep_cluster.clustering.kmeans import Kmeans
from deep_cluster.convnet.alexnet import AlexNet
from deep_cluster.preprocessing.dataset import Dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


def train_deep_clustering():
    """
    Main function that builds a dataset and a Deep clustering model based on Alexnet and PCA + K-means.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dicom_path = os.path.join(dir_path, "../../data/stage_2_train_images")

    nb_classes = 3
    learning_rate = 0.001

    dataset = Dataset(dicom_path, batch_size=128)
    logging.info("successfully loaded preprocessing")
    tf_dataset = dataset.get_train_dataset()

    alexnet = AlexNet(nb_classes=nb_classes)

    optimizer = tf.optimizers.Adam(learning_rate)
    kmeans_tf = Kmeans(k=nb_classes)

    loss_evolution = []
    accuracy_evolution = []

    i = 0
    for images in tf_dataset:
        logging.info("batch {}".format(i))
        output = alexnet.model(images, get_last_layer=False)

        pca = PCA(n_components=32, whiten=True)
        pca_transformed = pca.fit_transform(output)
        row_sums = np.linalg.norm(pca_transformed, axis=1)
        pca_transformed /= row_sums[:, np.newaxis]

        clusters_tf = kmeans_tf.fit_transform(pca_transformed)
        classification_output = tf.one_hot(clusters_tf, depth=nb_classes)
        loss = alexnet.train_step(inputs=images, outputs=classification_output, optimizer=optimizer)
        loss_evolution.append(loss)

        prediction = alexnet.model(images, get_last_layer=True)
        predicted_classes = tf.argmax(prediction, axis=1)
        equality = tf.math.equal(predicted_classes, clusters_tf)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
        accuracy_evolution.append(accuracy)
        logging.info("Accuracy: {}".format(accuracy))

        i += 1

    plt.figure()
    plt.plot(loss_evolution)
    plt.title("Loss evolution")
    plt.show()

    plt.figure()
    plt.plot(accuracy_evolution)
    plt.title("Train accuracy evolution")
    plt.show()


if __name__ == '__main__':
    train_deep_clustering()
