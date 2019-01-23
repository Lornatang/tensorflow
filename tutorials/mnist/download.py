import os

import urllib.request
import tensorflow as tf

# Download dataset
SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
WORK_DIRECTORY = './raw_data/mnist'


def download(filename):
    """Download the raw_data from Yann's website, unless it"""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(
            SOURCE_URL + filepath, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print(f"Successfully downloaded {filename} {size} bytes")
    return filepath
