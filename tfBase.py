import tensorflow as tf
print(tf.__version__)

import numpy as np 
import matplotlib
import sklearn

import pandas as pd 
import keras
import pandas as pd
from sklearn.metrics import confusion_matrix

print(sklearn.__version__)
print(keras.__version__)

h = tf.constant("Hello")
w = tf.constant(" World!")
hw = h + w

with tf.Session() as sess:
    ans = sess.run(hw)
    s = sess.run(w)

print(ans)
print(s)

print(h)
print(hw)


import sys
from util.vis_utils import show_plot
from util.minst_input_data import read_data_sets
# This is where the MNIST data will be downloaded to. If you already have it on your 
# machine then set the path accordingly to prevent an extra download. 
DATA_DIR = '../datasets/minist/'

# Load data 
data = read_data_sets(DATA_DIR, one_hot=True)

print("Nubmer of training-set images: {}".format(len(data.train.images)))
print("Luckily, there are also {} matching labels.".format(len(data.train.labels)))

import matplotlib.pyplot as plt 




#show_plot(data.train.images[:1000])  
"""
N_IMAGES = 1000

# Cut out the center part of the image (Actual digit)
center_img = [img.reshape(28, 28)[8:22, 8:22].ravel() 
              for img in  data.train.images[:N_IMAGES]]

# Sort by digits
sorted_lbls = np.argsort(data.train.labels.argmax(axis=1)[:N_IMAGES])
center_img = np.array(center_img)[sorted_lbls]

# Plot 
plt.figure()
plt.imshow(np.logical_not(center_img).T, cmap='gray')
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.title("Each column is the center of an image, unrolled...")
plt.show()
"""
"""
# We start by building the model 
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)+b


cross_entropy = \
    tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))


NUM_STEPS = 1000
MINIBATCH_SIZE = 100

with tf.Session() as sess:

    # Train
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})

    # Test
    is_correct, acc = sess.run([correct_mask, accuracy], 
                               feed_dict={x: data.test.images, y_true: data.test.labels})
    # Here we use the fetches [y_true, y_pred] since those are the vars we will need to
    # construct the confusion matrix.
    y_true_vec, y_pred_vec = sess.run([y_true, y_pred],
                              feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(acc*100))
print(is_correct)


correct_ix = np.where(is_correct)[0]
correct_img = data.train.images[correct_ix]

N_IMAGES = 10
img = np.vstack([np.hstack([img.reshape(28, 28) 
                            for img in correct_img[np.random.choice(len(correct_ix), N_IMAGES)]])
                 for i in range(N_IMAGES)])
img = np.logical_not(img)

plt.figure()
plt.imshow(img, cmap='gray')
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.title("We got this bunch right!")

incorrect_ix = np.where(np.logical_not(is_correct))[0]
incorrect_img = data.train.images[incorrect_ix]

N_IMAGES = 10
img = np.vstack([np.hstack([img.reshape(28, 28) 
                            for img in incorrect_img[np.random.choice(len(incorrect_ix), N_IMAGES)]])
                 for i in range(N_IMAGES)])
img = np.logical_not(img)

plt.figure()
plt.imshow(img, cmap='gray')
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.title("...but didn't do so well with these ones!")


# confusion_matrix() requires the actual predictions, not the probability vectors, so we use
# .argmax(axis=1) to select the class with the largest probability.
conf_mat = confusion_matrix(y_true_vec.argmax(axis=1), y_pred_vec.argmax(axis=1))

# pd.DataFrame is used for the nice print format
print(pd.DataFrame(conf_mat))
"""

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


# The image input -- a 1X784 vector 
x = tf.placeholder(tf.float32, shape=[None, 784])

# The correct label
y_ = tf.placeholder(tf.float32, shape=[None, 10])


# With CNNs we have a spatial notion, and need the image in the correct shape!
x_image = tf.reshape(x, [-1, 28, 28, 1])

# First conv layer
conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
conv1_pool = max_pool_2x2(conv1)

# Second conv layer
conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)

# We flatten the rectangular image representation before the fully-connected part 
conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])
full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

# Dropout
keep_prob = tf.placeholder(tf.float32)
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

# Readout layer 
y_conv = full_layer(full1_drop, 10)


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

STEPS = 10000
MINIBATCH_SIZE = 50

mnist=data
sess =  tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(STEPS):
    batch = mnist.train.next_batch(MINIBATCH_SIZE)

    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    if i % 200 == 0:
        valid_accuracy = sess.run(accuracy, 
                                  feed_dict={x: mnist.validation.images, 
                                             y_: mnist.validation.labels,
                                             keep_prob: 1.0})
        print("step {}, validation accuracy {}".format(i, valid_accuracy))


# Split the test part into 10 equal segments 
X = mnist.test.images.reshape(10, 1000, 784)
Y = mnist.test.labels.reshape(10, 1000, 10)
test_accuracy = np.mean([sess.run(accuracy, feed_dict={x:X[i], y_:Y[i], 
                                                       keep_prob:1.0}) for i in range(10)])#keep_prob is dropout

print("test accuracy: {:.4f}%".format(test_accuracy*100.))
sess.close()


import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

def readfile(path):
    try:
        data = img.imread(path)
        return data
    except:
        return np.array([])


def displayimage(path):
    data = img.imread(path)
    plt.imshow(data)
    plt.show()
    return
    
if __name__ == '__main__':


    # create a tensorflow constant string
    hello = tf.constant('Hello World!')

    # run within a session and print
    with tf.Session() as session:
        print("Tensorflow version: " + tf.__version__)
        print(hello.eval())
