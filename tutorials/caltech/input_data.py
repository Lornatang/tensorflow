# input_data.py

"""
    We need to do sample and label classification of generated images of specified size,
    obtain the get_files input by neural network,
    and meanwhile input raw_data for batch processing
    in order to facilitate network training.
"""

import os
import numpy as np
import tensorflow as tf


input_dir = 'data'

airplane = []
airplane_label = []

car = []
car_label = []

face = []
face_label = []

motorbike = []
motorbike_label = []


def convert_string_to_float(filename, label):
    """read filename to float for train.
    
    Args:
        filename: The train file name.
        label:    The train file label.

    Returns:
        image:    image data.
        label:    image label.

    """
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, label


def get_files(file_dir):
    """Get all the image path names in the directory,
    store them in the corresponding list,
    label them and store them in the label list.

    Args:
        file_dir:   train data dir.

    Returns:
        train_data: The train data for tensor
        train_label:The train label for tensor

    """
    for file in os.listdir(file_dir + '/airplane'):
        airplane.append(file_dir + '/airplane' + '/' + file)
        airplane_label.append(0)
    for file in os.listdir(file_dir + '/car'):
        car.append(file_dir + '/car' + '/' + file)
        car_label.append(1)
    for file in os.listdir(file_dir + '/face'):
        face.append(file_dir + '/face' + '/' + file)
        face_label.append(2)
    for file in os.listdir(file_dir + '/motorbike'):
        motorbike.append(file_dir + '/motorbike' + '/' + file)
        motorbike_label.append(3)

    # Disarrange the generated image path and label List,
    # and combine (airplane, car, face, motorbike) into a List (img and lab).
    image_list = np.hstack((airplane, car, face, motorbike))
    label_list = np.hstack(
        (airplane_label,
         car_label,
         face_label,
         motorbike_label))

    # Shuffle the order
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # Convert all img and lab to list
    image = list(temp[:, 0])
    label = list(temp[:, 1])

    # all sample num
    sample_num = len(label)

    train_data = image[0:sample_num]
    train_label = label[0:sample_num]
    # convert str to int save to list
    train_label = [int(i) for i in train_label]

    return train_data, train_label


def next_batch(image, label, batch_size=64):
    """Set the batch size for the exercise.

    Args:
        image:      image data
        label:      image label
        batch_size: default=64

    Returns:
        image_batch:
        label_batch

    """
    # Convert type
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int64)
    
    # image = tf.io.read_file(image)
    # image = tf.image.decode_jpeg(image)

    # make an input queue
    data = tf.data.Dataset.from_tensor_slices((image, label))
    # old func will remove
    # input_queue = tf.train.slice_input_producer([image, label])
    data = data.map(convert_string_to_float)

    data_batch = data.batch(batch_size)
    iterator = data_batch.make_one_shot_iterator()

    # generator batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int64
    with tf.Session() as sess:
        # sess.run(tf.initializers.global_variables())
        train_batch = sess.run(iterator.get_next())
        return train_batch
