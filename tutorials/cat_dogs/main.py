import os
import tensorflow as tf
import input_data
import model
import numpy as np

# parameters
N_LABELS = 2
IMG_W = 224  # img size
IMG_H = 224
BATCH_SIZE = 32
MAX_STEP = 1000
LEARNING_RATE = 0.0001

# define train data dir and logs dir
WORK_DIRECTORY = 'data'
LOGS_DIRECTORY = 'logs'


def main(_):
    # train raw_data and label
    train_data, train_label = input_data.get_files(WORK_DIRECTORY)
    train_batch, train_label_batch = input_data.next_batch(
        train_data,
        train_label
    )

    # define train op
    train_logits = model.inference(train_batch, N_LABELS, BATCH_SIZE)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.optimization(train_loss, LEARNING_RATE)
    train_acc = model.evaluation(train_logits, train_label_batch)

    # start logs
    summary_op = tf.summary.merge_all()

    # Save logs
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # writen to logs
    train_writer = tf.summary.FileWriter(LOGS_DIRECTORY, sess.graph)

    # queue monitor
    coord = tf.train.Coordinator()

    # train
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            # start op node
            _, loss, accuracy = sess.run([train_op, train_loss, train_acc])

            # print and write to logs.
            if step % 2 == 0:
                print(
                    f"Step [{step}/{MAX_STEP}] Loss {loss:.6f} Accuracy {accuracy * 100.0:.2f}%")
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            if accuracy >= 0.999:
                # Save logs
                checkpoint_path = os.path.join(
                    LOGS_DIRECTORY, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                break
        print(f"Model saved! Global step = {step}")

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')

    finally:
        coord.request_stop()
        sess.close()


if __name__ == '__main__':
    tf.app.run(main=main)
