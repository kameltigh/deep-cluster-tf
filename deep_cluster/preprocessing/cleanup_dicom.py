import collections
import logging
import os

from pydicom import dcmread


def __check_zero_pixels(file_path, patient_id, zero_ratio_threshold=.35):
    dcm_sample = dcmread(file_path)
    pixel_array = dcm_sample.pixel_array.flatten()
    zero_ratio = collections.Counter(pixel_array)[0] / len(pixel_array)

    if zero_ratio >= zero_ratio_threshold:
        logging.debug("image of the patient {} contains too many zero pixels".format(patient_id))
        return True
    return False


def remove_non_centered_images(dcm_dir):
    for file_path in os.listdir(dcm_dir):

        patient_id = file_path.split(".")[0]
        absolute_path = os.path.join(dcm_dir, file_path)
        to_remove = __check_zero_pixels(absolute_path, patient_id)
        if to_remove:
            logging.info("Removing file {}".format(absolute_path))
            os.remove(absolute_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    dir_path = os.path.dirname(os.path.abspath(__file__))
    dicom_path = os.path.join(dir_path, "../../data/stage_2_train_images")

    logging.info("removing non centered images from directory {}".format(dicom_path))

    remove_non_centered_images(dicom_path)
