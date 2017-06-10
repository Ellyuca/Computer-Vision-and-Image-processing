import argparse
import sys
from tensorflow.python.framework import dtypes
import tensorflow as tf
import numpy as np
from collections import namedtuple
import json
from os import makedirs
from os import path

FLAGS = None

Datasets = namedtuple('Datasets', ['train', 'validation', 'test'])


def export_def_graph(outdir="log"):
    if not path.exists(outdir):
        makedirs(outdir)
    writer = tf.summary.FileWriter(outdir, tf.get_default_graph())
    writer.close()
    print("+ Graph exported")


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class DataSet(object):

    def __init__(self,
                 images,
                 labels,
                 one_hot=False,
                 dtype=dtypes.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32, dtypes.float64):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        elif dtype == dtypes.float64:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float64)
            images = np.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_coil_data_sets(dtype=dtypes.float32):

    with open("coil_dataset_report.json", "r") as report_file:
        report = json.load(report_file)

    with open("coil_train_data.bin", "rb") as train_data_file:
        train_images = np.frombuffer(
            train_data_file.read(), dtype=np.float32)
        train_images = train_images.reshape(*report['train_data_shape'])

    with open("coil_train_labels.bin", "rb") as train_labels_file:
        train_labels = np.frombuffer(
            train_labels_file.read(), dtype=np.uint8)
        train_labels = train_labels.reshape(*report['train_labels_shape'])

    with open("coil_test_data.bin", "rb") as test_data_file:
        test_images = np.frombuffer(
            test_data_file.read(), dtype=np.float32)
        test_images = test_images.reshape(*report['test_data_shape'])

    with open("coil_test_labels.bin", "rb") as test_labels_file:
        test_labels = np.frombuffer(
            test_labels_file.read(), dtype=np.uint8)
        test_labels = test_labels.reshape(*report['test_labels_shape'])

    VALIDATION_SIZE = 200

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]

    train = DataSet(train_images, train_labels, dtype=dtype)
    validation = DataSet(validation_images, validation_labels, dtype=dtype)
    test = DataSet(test_images, test_labels, dtype=dtype)

    return Datasets(train=train, validation=validation, test=test)


class MyModel(object):

    def __init__(self):
        attributes = [
            "x",
            "y_",
            "W_conv1",
            "b_conv1",
            "x_image",
            "h_conv1",
            "h_pool1",
            "W_conv2",
            "b_conv2",
            "h_conv2",
            "h_pool2",
            "W_fc1",
            "b_fc1",
            "h_pool2_flat",
            "h_fc1",
            "keep_prob",
            "h_fc1_drop",
            "W_fc2",
            "b_fc2",
            "y_conv",
            "cross_entropy",
            "train_step",
            "correct_prediction",
            "accuracy",
            "variables"
        ]
        for attr in attributes:
            setattr(self, attr, None)


def gen_model():

    model = MyModel()

    print("+ Create model")
    with tf.name_scope('Conv_1'):
        # Create the model
        model.x = tf.placeholder(tf.float32, [None, 16384])

        # # Define loss and optimizer
        model.y_ = tf.placeholder(tf.float32, [None, 20])

        model.W_conv1 = weight_variable([5, 5, 1, 32])
        model.b_conv1 = bias_variable([32])

        # 128x128 images
        model.x_image = tf.reshape(model.x, [-1, 128, 128, 1])

        model.h_conv1 = tf.nn.relu(
            conv2d(model.x_image, model.W_conv1) + model.b_conv1)
        model.h_pool1 = max_pool_2x2(model.h_conv1)

    with tf.name_scope('Conv_2'):
        model.W_conv2 = weight_variable([5, 5, 32, 64])
        model.b_conv2 = bias_variable([64])

        model.h_conv2 = tf.nn.relu(
            conv2d(model.h_pool1, model.W_conv2) + model.b_conv2)
        model.h_pool2 = max_pool_2x2(model.h_conv2)

    with tf.name_scope('Full_Connected_1'):
        model.W_fc1 = weight_variable([32 * 32 * 64, 1024])
        model.b_fc1 = bias_variable([1024])

        # 3276800 / 32 / 32 / 64 = 50
        model.h_pool2_flat = tf.reshape(model.h_pool2, [-1, 32*32*64])
        model.h_fc1 = tf.nn.relu(
            tf.matmul(model.h_pool2_flat, model.W_fc1) + model.b_fc1)

        model.keep_prob = tf.placeholder(tf.float32)
        model.h_fc1_drop = tf.nn.dropout(model.h_fc1, model.keep_prob)

    with tf.name_scope('Full_Connected_2'):
        model.W_fc2 = weight_variable([1024, 20])
        model.b_fc2 = bias_variable([20])

        model.y_conv = tf.matmul(model.h_fc1_drop, model.W_fc2) + model.b_fc2

    model.cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model.y_conv, labels=model.y_))
    model.train_step = tf.train.AdamOptimizer(
        1e-4).minimize(model.cross_entropy)
    model.correct_prediction = tf.equal(
        tf.argmax(model.y_conv, 1), tf.argmax(model.y_, 1))
    model.accuracy = tf.reduce_mean(
        tf.cast(model.correct_prediction, tf.float32))

    model.variables = [
        model.W_conv1,
        model.b_conv1,
        model.W_conv2,
        model.b_conv2,
        model.W_fc1,
        model.b_fc1,
        model.W_fc2,
        model.b_fc2
    ]

    return model


def main(unparsed_args):
    # Import data
    print("+ Load data")
    coil = read_coil_data_sets()

    model = gen_model()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    export_def_graph()

    saver = tf.train.Saver(model.variables)

    if FLAGS.session is None:
        print("+ Train model")

        for i in range(2000):
            batch = coil.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = model.accuracy.eval(feed_dict={
                    model.x: batch[0], 
                    model.y_: batch[1], 
                    model.keep_prob: 1.0})
                print("+ Step %d, training accuracy %g" % (i, train_accuracy))
            print("+ Step %d" % i, end="\r")
            model.train_step.run(
                feed_dict={model.x: batch[0], 
                model.y_: batch[1], 
                model.keep_prob: 0.5})
        print("+ Save model")
        if not path.exists("models"):
            makedirs("models")
        saver.save(sess, "models/coil.chk")
    else:
        print("+ Load model")
        saver.restore(sess, FLAGS.session)

    print("+ Test accuracy %g" % model.accuracy.eval(feed_dict={
        model.x: coil.test.images, 
        model.y_: coil.test.labels, 
        model.keep_prob: 1.0}))


if __name__ == '__main__':
    # python coil_conv.py --session models\coil.chk
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', type=str, help='Previously saved session')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
