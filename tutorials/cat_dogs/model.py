import tensorflow as tf
import utils


def inference(images, classes, batch_size):
    """Build the AlexNet logs.

    Args:
      batch_size: The batch size is 32.
      images: Images Tensor
      classes: Image classes

    Returns:
      out: prediction image label.

    """
    # conv1
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [11, 11, 3, 64]), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1,
                                         shape=[64],
                                         dtype=utils.data_type()),
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        utils.print_activations(conv1)

    # lrn1
    with tf.name_scope('lrn1') as scope:
        lrn1 = tf.nn.lrn(conv1,
                         alpha=1e-4,
                         beta=0.75,
                         depth_radius=2,
                         bias=2.0
                         )

    # pool1
    pool1 = tf.nn.max_pool(lrn1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')
    utils.print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [5, 5, 64, 192]), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1,
                                         shape=[192],
                                         dtype=utils.data_type()),
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
    utils.print_activations(conv2)

    # lrn2
    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.lrn(conv2,
                         alpha=1e-4,
                         beta=0.75,
                         depth_radius=2,
                         bias=2.0)

    # pool2
    pool2 = tf.nn.max_pool(lrn2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')
    utils.print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [3, 3, 192, 384]), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1,
                                         shape=[384],
                                         dtype=utils.data_type()),
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        utils.print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [3, 3, 384, 256]), name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1,
                                         shape=[256],
                                         dtype=utils.data_type()),
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        utils.print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [3, 3, 256, 256]), name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1,
                                         shape=[256],
                                         dtype=utils.data_type()),
                             name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        utils.print_activations(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool5')
    utils.print_activations(pool5)

    # fully 1
    with tf.name_scope('fc1') as scope:
        reshape = tf.reshape(pool5, [batch_size, -1])
        dim = reshape.get_shape()[1].value
        kernel = tf.Variable(tf.truncated_normal(
            [dim, 4096]), name='weights')
        biases = tf.Variable(tf.constant(0.1,
                                         shape=[4096],
                                         dtype=utils.data_type()),
                             name='biases')
        fc1 = tf.nn.relu(tf.matmul(reshape, kernel) + biases)
        # dropout
        fc1 = tf.nn.dropout(fc1, 0.5)

    # fully 2
    with tf.name_scope('fc2') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [4096, 4096]), name='weights')
        biases = tf.Variable(tf.constant(0.1,
                                         shape=[4096],
                                         dtype=utils.data_type()),
                             name='biases')
        fc2 = tf.nn.relu(tf.matmul(fc1, kernel) + biases)
        # dropout
        fc2 = tf.nn.dropout(fc2, 0.5)

    # fully 3
    with tf.name_scope('fc3') as scope:
        kernel = tf.Variable(tf.truncated_normal(
            [4096, classes]), name='weights')
        biases = tf.Variable(tf.constant(0.1,
                                         shape=[classes],
                                         dtype=utils.data_type()),
                             name='biases')
        out = tf.add(tf.matmul(fc2, kernel), biases)

    return out


def losses(logits, labels):
    """Calculate the loss.

    Args:
        logits: prediction value.
        labels: real value.

    Returns:
        loss:   loss.

    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def optimization(loss, learning_rate):
    """optimizer loss value.

    Args:
        loss:           loss.
        learning_rate:  learning_rate.

    Returns:
        optimizer:      optimizer loss value.

    """
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(
            loss, global_step=global_step)
    return optimizer


def evaluation(logits, labels):
    """Evaluation/accuracy calculation.

    Args:
        logits:     prediction value.
        labels:     real value.

    Returns:
        accuracy:   Average accuracy of current step.
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
