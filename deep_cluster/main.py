import logging
import os

from deep_cluster.clustering.kmeans import Kmeans as tf_kmeans
from deep_cluster.clustering.pca import PCA as tf_pca
from deep_cluster.convnet.alexnet import AlexNet
from deep_cluster.preprocessing.dataset import Dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))

    dataset = Dataset(os.path.join(dir_path, "../../data/stage_2_train_images"))
    logging.info("successfully loaded preprocessing")
    tf_dataset = dataset.get_train_dataset()

    alexnet = AlexNet()

    tf_dataset.map(alexnet.build_model)

    for image in tf_dataset.take(1):
        output = alexnet.build_model(image)
        pca_tf = tf_pca(output, k=32)
        kmeans_tf = tf_kmeans(k=50)
        clusters_tf = kmeans_tf.fit_transform(pca_tf.y.numpy())
        logging.info(clusters_tf.shape)
