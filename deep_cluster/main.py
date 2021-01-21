import logging
import os

import tensorflow as tf

from deep_cluster.clustering.kmeans import Kmeans as tf_kmeans
from deep_cluster.clustering.pca import PCA as tf_pca
from deep_cluster.convnet.alexnet import AlexNet
from deep_cluster.preprocessing.dataset import Dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))

    nb_classes = 3
    learning_rate = 0.001

    dataset = Dataset(os.path.join(dir_path, "../../data/stage_2_train_images"), batch_size=128)
    logging.info("successfully loaded preprocessing")
    tf_dataset = dataset.get_train_dataset()

    alexnet = AlexNet(nb_classes=nb_classes)

    optimizer = tf.optimizers.Adam(learning_rate)
    kmeans_tf = tf_kmeans(k=nb_classes)

    for image in tf_dataset:
        output = alexnet.model(image, get_last_layer=False)
        pca_tf = tf_pca(output, k=16)
        clusters_tf = kmeans_tf.fit_transform(pca_tf.y.numpy())
        classification_output = tf.one_hot(clusters_tf, depth=nb_classes)
        alexnet.train_step(inputs=image, outputs=classification_output, optimizer=optimizer)
