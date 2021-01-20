import logging
import os
import matplotlib.pyplot as plt
from deep_cluster.preprocessing.dataset import Dataset
from deep_cluster.convnet.alexnet import AlexNet
import tensorflow as tf

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))

    dataset = Dataset(os.path.join(dir_path, "../../data/stage_2_train_images"))
    logging.info("successfully loaded preprocessing")
    tf_dataset = dataset.get_train_dataset()

    alexnet = AlexNet()



    for image in tf_dataset.take(1):
        plt.figure()
        plt.imshow(tf.squeeze(image[0]).numpy())
        plt.show()
        print(alexnet.build_model(image).shape)
