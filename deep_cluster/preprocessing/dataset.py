import os

import tensorflow as tf
import tensorflow_io as tfio


class Dataset:
    DATASET_SIZE = 2000
    IMAGE_SIZE = 227
    BATCH_SIZE = 32
    PREFETCH_SIZE = 32

    def __init__(self, dicom_path: str):
        list_ds = tf.data.Dataset.list_files(os.path.join(dicom_path, "*.dcm"), shuffle=False)
        list_ds = list_ds.shuffle(self.DATASET_SIZE, reshuffle_each_iteration=False)

        val_size = int(self.DATASET_SIZE * 0.2)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)

        self.train_ds = train_ds.map(Dataset.__load_image, num_parallel_calls=32).batch(Dataset.BATCH_SIZE).prefetch(
            Dataset.PREFETCH_SIZE)
        self.val_ds = val_ds.map(Dataset.__load_image, num_parallel_calls=32).batch(Dataset.BATCH_SIZE).prefetch(
            Dataset.PREFETCH_SIZE)

    @staticmethod
    def __load_image(path):
        dcm_img = tf.io.read_file(path)
        img = tfio.image.decode_dicom_image(dcm_img)
        return tf.squeeze(tf.image.resize(img, [Dataset.IMAGE_SIZE, Dataset.IMAGE_SIZE]), [0])

    def get_train_dataset(self) -> tf.data.Dataset:
        return self.train_ds
