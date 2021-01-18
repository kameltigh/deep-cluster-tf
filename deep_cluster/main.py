import logging
import os

from deep_cluster.dataset.dataset import Dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

dir_path = os.path.dirname(os.path.abspath(__file__))

dataset = Dataset(os.path.join(dir_path, "../../data/stage_2_train_images"))
logging.info("successfully loaded dataset")
tf_dataset = dataset.get_train_dataset()

for image in tf_dataset:
    print(image.numpy())
