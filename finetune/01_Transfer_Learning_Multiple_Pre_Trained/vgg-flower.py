# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.misc import imread, imresize
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import tensorflow as tf
import keras 
from keras.applications.vgg16 import VGG16
from keras.applications import VGG19, InceptionV3, Xception, ResNet50
from keras.optimizers import SGD
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from keras.applications.imagenet_utils import preprocess_input as preprocess_type1
from keras.applications.inception_v3 import preprocess_input as preprocess_type2
from cub_util import CUB200
#from mlxtend.classifier import StackingClassifier
#from mlxtend.feature_selection import ColumnSelector
import matplotlib.pyplot as plt 


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


DATA_DIR = '../../datasets/mnist'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100


data = input_data.read_data_sets(DATA_DIR, one_hot=True)
X=data.test.images
Y=data.test.labels
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y_pred, labels=y_true))

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
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))





"""
DATA_DIR = '/tmp/data' if not 'win' in sys.platform else "c:\\tmp\\data"
IMAGE_DIR = os.path.join(DATA_DIR, "flowers")
DEFAULT_VGG_IMAGE_SIZE = (224, 224)
NUM_IMAGES = 8141
NUM_CLASSES = 102
model = VGG16(include_top=False, weights='imagenet')

#获取特征图
def load_features_compute_once(model, im_size, preprocess, save_path):
        
    if os.path.exists(save_path):
        data = pd.read_csv(save_path, compression='gzip', header=0, index_col=0)
        X = data.values 
        y = data.index.values
    else:
        X, y = CUB200(CUB_DIR, image_size=im_size).load_dataset()
        X = model(include_top=False, weights="imagenet", pooling='avg').predict(preprocess(X))
        pd.DataFrame(X, index=y).to_csv(save_path, compression='gzip', header=True, index=True)

    return X, y


def load_single_img(file_name, resize_to=DEFAULT_VGG_IMAGE_SIZE):
    img = imread(os.path.join(IMAGE_DIR, "jpg", file_name))
    img = imresize(img, resize_to)
    return img


def load_images_labels(use_classes=None, resize_to=DEFAULT_VGG_IMAGE_SIZE):
    
    # Load the .mat label file
    labels = loadmat(os.path.join(IMAGE_DIR, "imagelabels.mat"))["labels"].ravel()

    # If use_classes is None, it becomes all 102 available classes
    use_classes = use_classes or list(range(NUM_CLASSES))

    # Compile a list of flower-image files we are going to use, and the associated label in the format [(file, label),
    file_name_label = [("image_{:05}.jpg".format(i+1), labels[i])
                       for i in range(NUM_IMAGES) if labels[i] in use_classes]

    # Load images and labels
    images = [load_single_img(file_name, resize_to=resize_to) for file_name, _ in file_name_label]
    images = np.array(images)
    labels = [l for _, l in file_name_label]

    return images, labels


NUM_CLASSES = 200
DATA_DIR = os.path.expanduser(os.path.join("~", "data", "blog")) 
CUB_DIR = os.path.join(DATA_DIR, "CUB_200_2011", "images")
FEATURES_DIR = os.path.join(DATA_DIR, "features")

# If this fails, the CUB-200 dataset is not in the rigt place... :) 
assert os.path.exists(CUB_DIR)
#load data and show
X, _ = CUB200(CUB_DIR, image_size=(100, 100)).load_dataset(num_per_class=5)
n = X.shape[0]
rnd_birds = np.vstack([np.hstack([X[np.random.choice(n)] for i in range(10)])
                       for j in range(10)])
plt.figure(figsize=(8, 8))
plt.imshow(rnd_birds / 255)
plt.gca().get_xaxis().set_visible(False)
plt.gca().get_yaxis().set_visible(False)
plt.title("100 random birds...", fontsize=30)

X_resnet, y = load_features_compute_once(ResNet50, (244, 244), preprocess_type1, 
                                         os.path.join(FEATURES_DIR, "CUB200_resnet"))

clf = LinearSVC()
results = cross_val_score(clf, X_resnet, y, cv=3, n_jobs=-1)


X_resnet, y = load_features_compute_once(ResNet50, (244, 244), preprocess_type1, 
                                         os.path.join(FEATURES_DIR, "CUB200_resnet"))

X_vgg, _ = load_features_compute_once(VGG19, (244, 244), preprocess_type1, 
                                         os.path.join(FEATURES_DIR, "CUB200_VGG19"))

X_incept, _ = load_features_compute_once(InceptionV3, (299, 299), preprocess_type2, 
                                         os.path.join(FEATURES_DIR, "CUB200_inception"))

X_xcept, _ = load_features_compute_once(Xception, (299, 299), preprocess_type2, 
                                         os.path.join(FEATURES_DIR, "CUB200_xception"))


X_all = np.hstack([X_vgg, X_resnet, X_incept, X_xcept])
inx = np.cumsum([0] + [X_vgg.shape[1], X_resnet.shape[1], X_incept.shape[1], X_xcept.shape[1]])

y = LabelEncoder().fit_transform(y)

base_classifier = LogisticRegression
meta_classifier = LinearSVC

pipes = [make_pipeline(ColumnSelector(cols=list(range(inx[i], inx[i+1]))), base_classifier())
         for i in range(4)]

stacking_classifier = StackingClassifier(classifiers=pipes, meta_classifier=meta_classifier(), 
                                         use_probas=True, average_probas=True, verbose=1)

results = cross_val_score(stacking_classifier, X_all, y, cv=3, n_jobs=-1)

print(results)
print("Overall accuracy: {:.3}".format(np.mean(results) * 100.))
"""