from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

from util.layer import conv_layer, max_pool_2x2, full_layer

from util.minst_input_data import read_data_sets
MINIBATCH_SIZE = 50
STEPS = 5000

DATA_DIR = '../datasets/minist/'

# Load data 
mnist  = read_data_sets(DATA_DIR, one_hot=True)


x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])
conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        batch = mnist.train.next_batch(MINIBATCH_SIZE)

        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1],
                                                           keep_prob: 1.0})
            print("step {}, training accuracy {}".format(i, train_accuracy))

        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean(
        [sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], keep_prob: 1.0}) for i in range(10)])

print("test accuracy: {}".format(test_accuracy))


"""
We construct a confusion matrix for the softmax regression on MNIST digits.
1. Model is built and run like before.
2. During the test phase we use y_true, y_pred in the fetch argument, since these
are the vars we will need for the confusion matrix.
3. We then use the built-in confusion_matrix method in sklearn.metrics to complete
the task.
"""

"""
import sys
import tensorflow as tf
from util.minst_input_data import read_data_sets
import pandas as pd
from sklearn.metrics import confusion_matrix
DATA_DIR = '../datasets/minist/'

# Load data 
data = read_data_sets(DATA_DIR, one_hot=True)

print("Nubmer of training-set images: {}".format(len(data.train.images)))
print("Luckily, there are also {} matching labels.".format(len(data.train.labels)))



# This is where the MNIST data will be downloaded to. If you already have it on your
# machine then set the path accordingly to prevent an extra download.
#DATA_DIR = '/tmp/data' if not 'win' in sys.platform else "c:\\tmp\\data"



NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# We start by building the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)
#y_pred = tf.matmul(x, W) + b

cross_entropy = \
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:

    # Train
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    # Here we use the fetches [y_true, y_pred] since those are the vars we will need to
    # construct the confusion matrix.
    y_true_vec, y_pred_vec = sess.run([y_true, y_pred],
                              feed_dict={x: data.test.images, y_true: data.test.labels})

# confusion_matrix() requires the actual predictions, not the probability vectors, so we use
# .argmax(axis=1) to select the class with the largest probability.
conf_mat = confusion_matrix(y_true_vec.argmax(axis=1), y_pred_vec.argmax(axis=1))

# pd.DataFrame is used for the nice print format
print(pd.DataFrame(conf_mat))

"""





"""
We add a bias term to the regression model. To do so we need to change only 2 lines of code.
First, we define a bias variable (one for each of the 10 digits):
b = tf.Variable(tf.zeros([10]))

Next, we add it to the model:
y_pred = tf.matmul(x, W) + b

So the model is now:
    y_pred(i) = <x, w_i> + b_i,
and in matrix form:
    y_pred = Wx + b
"""
"""
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import pandas as pd
from sklearn.metrics import confusion_matrix

from util.minst_input_data import read_data_sets


DATA_DIR = '../datasets/minist/'
# This is where the MNIST data will be downloaded to. If you already have it on your
# machine then set the path accordingly to prevent an extra download.


# Load data
data = read_data_sets(DATA_DIR, one_hot=True)

NUM_STEPS = 1000
MINIBATCH_SIZE = 100

# We start by building the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W) + b

cross_entropy = \
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

with tf.Session() as sess:

    # Train
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    is_correct, acc = sess.run([correct_mask, accuracy],
                               feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(acc*100))

"""
