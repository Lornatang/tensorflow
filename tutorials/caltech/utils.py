import os
import tensorflow as tf


def sum_of_file(dirpath):
    """sum of images.

    Args:
        dirpath: input dir

    Returns:
        image num

    """
    num = 0
    for dir in os.listdir(dirpath):
        for _ in os.listdir(dirpath + '/' + dir):
            num += 1
    return num


def print_activations(t):
    print(t.op.name, f" {t.get_shape().as_list()}")


def data_type():
    return tf.float32
