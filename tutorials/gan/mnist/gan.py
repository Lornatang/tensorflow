import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


WORK_DIRECTORY = './data/mnist'
IMAGE_SIZE = 28
NUM_CHANNELS = 1  # black and white no rgb.
PIXEL_DEPTH = 255
SEED = 66487
BATCH_SIZE = 128
NUM_EPOCHS = 10
Z_dim = 100

FLAGS = None


def data_type():
    "Return the type of the activations, weights, and placeholder variables."""
    return tf.float32


"""generator para"""
Z = tf.placeholder(data_type(), shape=[BATCH_SIZE, Z_dim])

G_W1 = tf.Variable(tf.constant_initializer([Z_dim, BATCH_SIZE]))
G_b1 = tf.Variable(tf.zeros(shape=[BATCH_SIZE]))

G_W2 = tf.Variable(tf.constant_initializer([BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE]))
G_b2 = tf.Variable(tf.zeros(shape=[IMAGE_SIZE * IMAGE_SIZE]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(Z):
    G_h1 = tf.nn.relu(tf.matmul(Z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


"""discriminator para"""
X = tf.placeholder(data_type(), shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

D_W1 = tf.Variable(tf.constant_initializer([IMAGE_SIZE * IMAGE_SIZE, BATCH_SIZE]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(tf.constant_initializer([BATCH_SIZE, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def discriminator(X):
    D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                                     labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                     labels=tf.zeros_like(D_logit_fake)))
D_loss = D_logit_real + D_logit_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)


# def plot(samples):
#     fig = plt.figure(figsize=(4, 4))
#     gs = gridspec.GridSpec(4, 4)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
#
#     return fig
#
#
#
# if not os.path.exists('out/'):
#     os.makedirs('out/')
#
# sess.run(tf.global_variables_initializer())
#
# i = 0
# for it in range(1000000):
#     if it % 1000 == 0:
#         samples = sess.run(G_sample, feed_dict={
#                            Z: sample_Z(16, Z_dim)})  # 16*784
#         fig = plot(samples)
#         plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
#         i += 1
#         plt.close(fig)
#
#     X_mb, _ = mnist.train.next_batch(mb_size)
#
#     _, D_loss_curr = sess.run([D_optimizer, D_loss], feed_dict={
#                               X: X_mb, Z: sample_Z(mb_size, Z_dim)})
#     _, G_loss_curr = sess.run([G_optimizer, G_loss], feed_dict={
#                               Z: sample_Z(mb_size, Z_dim)})
#
#     if it % 1000 == 0:
#         print('Iter: {}'.format(it))
#         print('D loss: {:.4}'.format(D_loss_curr))
#         print('G_loss: {:.4}'.format(G_loss_curr))
#         print()

