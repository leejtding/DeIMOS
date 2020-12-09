import numpy as np
import tensorflow as tf
import utils
import glob
import os


def get_data(data_dir):
    """
    Given a data directory, returns two tensorflow Dataset objects for the labelled and unlabelled images. The labelled
    Dataset contains tuples of image tensors and class labels, while the unlabelled Dataset just contains images.
    :param data_dir: The directory of the data, e.g. 'data/hirise-map-proj-v3_2'
    :return: Two Dataset objects.
    """
    labelled_path = glob.glob('./' + data_dir + '/labeled/*.jpg')
    unlabelled_path = glob.glob('./' + data_dir + '/unlabeled/*.jpg')
    label_path = data_dir + '/labels-map-proj_v3_2.txt'  # TODO abstract these file paths? probably fine as is

    labels_dict = utils.get_class_dict(label_path)

    labelled_set = tf.data.Dataset.list_files(labelled_path)
    unlabelled_set = tf.data.Dataset.list_files(unlabelled_path)

    def process_labelled(file_path):
        file_name = tf.strings.split(file_path, os.sep)[-1].numpy().decode("utf-8")
        image = tf.cast(
            tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(file_path)), [227, 227]), tf.float32) / 255
        label = labels_dict[file_name] - 1 # Convert class 1-7 to 0-6
        return image, label

    labelled_data = labelled_set.map(
        lambda x: tf.py_function(func=process_labelled, inp=[x], Tout=(tf.float32, tf.int32)))

    def process_unlabelled(file_path):
        image = tf.cast(
            tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(file_path)), [227, 227]), tf.float32) / 255
        return image

    unlabelled_data = unlabelled_set.map(process_unlabelled)
    return labelled_data, unlabelled_data
