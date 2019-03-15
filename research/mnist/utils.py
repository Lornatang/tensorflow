import os
from PIL import Image
import tensorflow as tf


def create_record(path):
    cwd = os.getcwd()
    classes = os.listdir(cwd + path)

    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd + path + name + "/"
        print(class_path)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((32, 32))
                image = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name)])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }))
                writer.write(example.SerializeToString())
    writer.close()


create_record("/train/")
