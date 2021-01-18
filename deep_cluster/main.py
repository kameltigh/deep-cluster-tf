import logging
import os

from deep_cluster.preprocessing.dataset import Dataset
from deep_cluster.convnet.alexnet import AlexNet

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.abspath(__file__))

    dataset = Dataset(os.path.join(dir_path, "../../data/stage_2_train_images"))
    logging.info("successfully loaded preprocessing")
    tf_dataset = dataset.get_train_dataset()

    alexnet = AlexNet(tf_dataset)

    for image in tf_dataset:
        print(image.numpy())
