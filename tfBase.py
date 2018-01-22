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
from util.minst_input_data import read_data_sets
# This is where the MNIST data will be downloaded to. If you already have it on your 
# machine then set the path accordingly to prevent an extra download. 
DATA_DIR = '../datasets/minist/'

# Load data 
data = read_data_sets(DATA_DIR, one_hot=True)

print("Nubmer of training-set images: {}".format(len(data.train.images)))
print("Luckily, there are also {} matching labels.".format(len(data.train.labels)))

import matplotlib.pyplot as plt 


def show_plot(data,num=0, multi=False):
    if multi is True:
        N_IMAGES = num
        img = np.vstack([np.hstack([img.reshape(28, 28) 
                            for img in data.train.images[np.random.choice(1000, N_IMAGES)]])
                 for i in range(N_IMAGES)])
        img = np.logical_not(img)
        
        title="{} random digits".format(num*num)
    else:
        if num is 0:
            img = data    
            #img = data.train.images[:1000]
            img =np.logical_not(img).T
            title="Each column is an image unrolled..."
        else:
            IMAGE_IX_IN_DATASET = num
            img = data.train.images[IMAGE_IX_IN_DATASET].reshape(28, 28)
            lbl = data.train.labels[IMAGE_IX_IN_DATASET].argmax()
            title="This is supposed to be a {}".format(lbl)
            # Get the raw (vector) image and labael -- see what it looks like when not a rectangle       
    # Plot 
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    plt.title(title)
    plt.show()

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
