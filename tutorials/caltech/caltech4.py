# cat_dogs.py

"""Convert the original image to the required size and save it"""
import os
import tensorflow as tf
import cv2
import numpy as np

import utils

INPUT_DIR = 'raw_data/'
OUTPUT_DIR = 'data/'

LOGS_DIR = 'logs'

# The type of recognition required
CLASSES = {'airplane', 'car', 'face', 'motorbike'}
N_CLASSES = 4

# check output dir exists
if not tf.gfile.Exists(OUTPUT_DIR):
    tf.gfile.MakeDirs(OUTPUT_DIR)

# check logs dir exists
if not tf.gfile.Exists(LOGS_DIR):
    tf.gfile.MakeDirs(LOGS_DIR)

# create classes dir
for _, dir_name in enumerate(CLASSES):
    if not tf.gfile.Exists(OUTPUT_DIR + dir_name):
        tf.gfile.MakeDirs(OUTPUT_DIR + dir_name)


def create_record():
    """make TFRecords raw_data"""
    writer = tf.python_io.TFRecordWriter("caltech_4.tfrecords")
    for index, name in enumerate(CLASSES):
        if name == 'airplane':
            index = 0
        elif name == 'car':
            index = 1
        elif name == 'face':
            index = 2
        else:
            index = 3
        class_path = INPUT_DIR + "/" + name + "/"
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(
                img, (224, 224), interpolation=cv2.INTER_NEAREST)  # resize img
            data = img.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index]))
                }))

            writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename):
    """Read the image into the tensorFlow queue.

    Args:
        filename: read image to tf queue path.

    Returns:
        data: image raw_data.
        label:image label.

    """
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)
        })
    label = features['label']
    data = features['data']
    data = tf.decode_raw(data, tf.uint8)
    data = tf.reshape(data, [224, 224, 3])
    label = tf.cast(label, tf.int32)
    return data, label


def main(_):
    num_examples = utils.sum_of_file(INPUT_DIR)
    print(f"Images total: {num_examples}.\n")
    print(f"Start create record!\n")
    create_record()
    print(f"Record create successful.")
    batch = read_and_decode('caltech_4.tfrecords')
    # Distributed tensorflow
    init_op = tf.group(
        tf.initializers.local_variables(),
        tf.initializers.global_variables()
    )

    with tf.Session() as sess:
        # init all variables
        sess.run(init_op)
        print("Init all variables complete!")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(num_examples):
            example, lab = sess.run(batch)
            # Convert the array to an image
            img = cv2.cvtColor(
                np.asarray(example),
                cv2.COLOR_RGB2BGR)
            if lab == 0:
                classes = 'airplane'
            elif lab == 1:
                classes = 'car'
            elif lab == 2:
                classes = 'face'
            else:
                classes = 'motorbike'
            cv2.imwrite(
                OUTPUT_DIR +
                '/' +
                classes +
                '/' +
                str(i) +
                '.jpg',
                img)
        print(f"Image written to '{OUTPUT_DIR}'.")
        coord.request_stop()
        coord.join(threads)
        sess.close()


if __name__ == '__main__':
    tf.app.run(main=main)
