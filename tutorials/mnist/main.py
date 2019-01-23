# Copy from google tutorials
"""
  A test of mnist using the AlexNet logs
"""

import argparse
import sys
import time

from download import *
from tools import *

import numpy as np
import tensorflow as tf

# Download dataset
SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
WORK_DIRECTORY = './raw_data/mnist'
IMAGE_SIZE = 28
NUM_CHANNELS = 1  # black and white no rgb.
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # size of the validation set.
SEED = 66487
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


FLAGS = None


def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32


def error_rate(predictions, labels):
    """Return the error rate base on dense predictions"""
    return 100.00 - (100.00 * np.sum(np.argmax(predictions, 1) == labels) /
                     predictions.shape[0])


def main(_):
    # Get the raw_data.
    train_data_filename = download('train-images-idx3-ubyte.gz')
    train_labels_filename = download('train-labels-idx1-ubyte.gz')
    test_data_filename = download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS

    train_size = train_labels.shape[0]
    x = tf.placeholder(
        data_type(),
        shape=[
            BATCH_SIZE,
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS])
    y = tf.placeholder(tf.int64, shape=[BATCH_SIZE, ])
    eval_data = tf.placeholder(data_type(),
                               shape=(EVAL_BATCH_SIZE,
                                      IMAGE_SIZE,
                                      IMAGE_SIZE,
                                      NUM_CHANNELS))

    # The variables below hold all the trainable weights.
    conv1_weights = tf.Variable(tf.truncated_normal([11, 11, NUM_CHANNELS, 64],
                                                    stddev=0.1,
                                                    seed=SEED,
                                                    dtype=data_type()))
    conv1_biases = tf.Variable(
        tf.zeros([64], dtype=data_type()))

    conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                    stddev=0.1,
                                                    seed=SEED,
                                                    dtype=data_type()))
    conv2_biases = tf.Variable(
        tf.constant(
            0.1,
            shape=[192],
            dtype=data_type()))

    conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                    stddev=0.1,
                                                    seed=SEED,
                                                    dtype=data_type()))
    conv3_biases = tf.Variable(
        tf.constant(
            0.1,
            shape=[384],
            dtype=data_type()))

    conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                    stddev=0.1,
                                                    seed=SEED,
                                                    dtype=data_type()))
    conv4_biases = tf.Variable(
        tf.constant(
            0.1,
            shape=[256],
            dtype=data_type()))

    conv5_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                    stddev=0.1,
                                                    seed=SEED,
                                                    dtype=data_type()))
    conv5_biases = tf.Variable(
        tf.constant(
            0.1,
            shape=[256],
            dtype=data_type()))

    # fully connected, depth 1024
    fc1_weights = tf.Variable(tf.truncated_normal([1 * 1 * 256, 4096],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
    fc1_biases = tf.Variable(
        tf.constant(
            0.1,
            shape=[4096],
            dtype=data_type()))

    fc2_weights = tf.Variable(tf.truncated_normal([4096, 4096],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
    fc2_biases = tf.Variable(
        tf.constant(
            0.1,
            shape=[4096],
            dtype=data_type()))

    fc3_weights = tf.Variable(tf.truncated_normal([4096, NUM_LABELS],
                                                  stddev=0.1,
                                                  seed=SEED,
                                                  dtype=data_type()))
    fc3_biases = tf.Variable(
        tf.constant(
            0.1,
            shape=[10],
            dtype=data_type()))

    def model(data):
        """The logs definition"""
        # Conv 1
        with tf.name_scope('conv1'):
            conv1 = tf.nn.conv2d(data,
                                 conv1_weights,
                                 strides=[1, 4, 4, 1],
                                 padding='SAME')
            # Bias and rectified linear non_linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
            norm = tf.nn.local_response_normalization(relu,
                                                      depth_radius=2,
                                                      bias=2.0,
                                                      alpha=1e-4,
                                                      beta=0.75)
            print_activations(conv1)
        # Max pooling.The kernel size spec {ksize} also follows the layout.
        pool1 = tf.nn.max_pool(norm,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
        print_activations(pool1)

        # Conv 2
        with tf.name_scope('conv2'):
            conv2 = tf.nn.conv2d(pool1,
                                 conv2_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME')
            # Bias and rectified linear non_linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
            norm = tf.nn.local_response_normalization(relu,
                                                      alpha=1e-4,
                                                      beta=0.75,
                                                      depth_radius=2,
                                                      bias=2.0)
            print_activations(conv2)
        # Max pooling.The kernel size spec {ksize} also follows the layout.
        pool2 = tf.nn.max_pool(norm,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
        print_activations(pool2)

        # Conv 3
        with tf.name_scope('conv3'):
            conv3 = tf.nn.conv2d(pool2,
                                 conv3_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME')
            # Bias and rectified linear non_linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
            print_activations(conv3)

        # Conv 4
        with tf.name_scope('conv4'):
            conv4 = tf.nn.conv2d(relu,
                                 conv4_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME')
            # Bias and rectified linear non_linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
            norm = tf.nn.lrn(relu,
                             4,
                             bias=1.0,
                             alpha=0.001 / 9.0,
                             beta=0.75)
            print_activations(conv4)

        # Conv 5
        with tf.name_scope('conv5'):
            conv5 = tf.nn.conv2d(norm,
                                 conv5_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME')
            # Bias and rectified linear non_linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
            print_activations(conv5)
        # Max pooling.The kernel size spec {ksize} also follows the layout.
        pool5 = tf.nn.max_pool(relu,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
        print_activations(pool5)

        # Fully 1
        fc1 = tf.reshape(pool5, [-1, fc1_weights.get_shape().as_list()[0]])
        fc1 = tf.nn.relu(tf.matmul(fc1, fc1_weights) + fc1_biases)
        # dropout
        fc1 = tf.nn.dropout(fc1, 0.5)

        # Fully 2
        fc2 = tf.reshape(fc1, [-1, fc2_weights.get_shape().as_list()[0]])
        fc2 = tf.nn.relu(tf.matmul(fc2, fc2_weights) + fc2_biases)
        # dropout
        fc2 = tf.nn.dropout(fc2, 0.5)

        # Fully 3
        fc3 = tf.reshape(fc2, [-1, fc3_weights.get_shape().as_list()[0]])
        out = tf.nn.relu(tf.matmul(fc3, fc3_weights) + fc3_biases)
        return out

    # Training computation: logits + cross_entropy loss.
    logits = model(x)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) +
                    tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) +
                    tf.nn.l2_loss(fc2_biases) +
                    tf.nn.l2_loss(fc3_weights) +
                    tf.nn.l2_loss(fc3_biases))
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=data_type())
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        0.01,
        batch * BATCH_SIZE,
        train_size,
        0.95,
        staircase=True)
    # Use Adam for the optimization.
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.9).minimize(loss=loss, global_step=batch)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation.
    eval_prediction = tf.nn.softmax(model(eval_data))

    def eval_in_batch(session, data):
        """Get all predictions for a dataset by running."""
        size = data.shape[0]
        if size < EVAL_BATCH_SIZE:
            raise ValueError("Batch size for evals larges than dataset:")
        pred = np.ndarray(shape=(size, NUM_LABELS))
        for begin in range(0, size, EVAL_BATCH_SIZE):
            end = begin + EVAL_BATCH_SIZE
            if end <= size:
                pred[begin:end, :] = session.run(eval_prediction, feed_dict={
                    eval_data: data[begin:end, ...]})
            else:
                batch_predictions = session.run(eval_prediction,
                                                feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
                pred[begin:, :] = batch_predictions[begin - size:, :]
        return pred
    start_time = time.time()
    with tf.Session() as sess:
        # Run all the initializers
        tf.global_variables_initializer().run()
        print("Init all variables complete!")
        for step in range(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {x: batch_data,
                         y: batch_labels}
            sess.run(optimizer, feed_dict=feed_dict)
            if step % EVAL_FREQUENCY == 0:
                # fetch some extra node's raw_data
                l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                              feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print(f"Step {step} "
                      f"(epoch {(float(step) * BATCH_SIZE / train_size):.2f}) "
                      f"{(1000 * elapsed_time / EVAL_FREQUENCY):.1f} ms")
                print(f"Minibatch loss: {l:.3f}, learning rate: {lr:.6f}")
                print(
                    f"Minibatch error: {error_rate(predictions,batch_labels):.1f}%")

                print(
                    f"Validation error: {error_rate(eval_in_batch(sess, validation_data), validation_labels):.1f}%")
                sys.stdout.flush()
                # Finally print the result!
        test_error = error_rate(eval_in_batch(sess, test_data), test_labels)
        print(f"Test error: {test_error:.1f}%.")
        if FLAGS.self_test:
            print(f"test_error {test_error}")
            assert test_error == 0.0, f"expected 0.0 test_error, got {test_error:.2f}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--use_fp16',
        default=False,
        help='Use half floats instead of full floats if True.',
        action='store_true')
    parser.add_argument(
        '--self_test',
        default=False,
        action='store_true',
        help='True if running a self test.')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
