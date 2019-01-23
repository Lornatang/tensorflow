import gzip

import numpy as np

# Download dataset
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
BATCH_SIZE = 128


def extract_data(filename, num_images):
    """Extract the image into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print(f"Extracting {filename}")
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(
            IMAGE_SIZE *
            IMAGE_SIZE *
            num_images *
            NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print(f"Extracting {filename}")
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def print_activations(t):
    print(t.op.name, f" {t.get_shape().as_list()}")
