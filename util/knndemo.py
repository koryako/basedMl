import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import knn
from cs231n.data_utils import *
import operator
from cs231n.classifiers.k_nearest_neighbor import *
from cs231n.classifiers.linear_classifier import *


cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir) 
print 'load_CIFAR10'
num_training = 500
mask = range(num_training) 
X_train = X_train[mask]
y_train = y_train[mask] 
num_test = 500
mask = range(num_test) 
X_test = X_test[mask]
y_test = y_test[mask]
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
classifier = KNearestNeighbor() 
classifier.train(X_train, y_train)
dists = classifier.compute_distances_two_loops(X_test)
#dists_one = classifier.compute_distances_one_loop(X_test)
dists_no=classifier.compute_distances_no_loops(X_test)

difference = np.linalg.norm(dists - dists_no, ord='fro')
print 'Difference was: %f' % (difference, )
if difference < 0.001:
    print 'Good! The distance matrices are the same'
else:
    print 'Uh-oh! The distance matrices are different'
#y_test_pred = classifier.predict_labels(dists, k=1)
#num_correct = np.sum(y_test_pred == y_test)
#accuracy = float(num_correct) / num_test
#print 'Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy)#/


def time_function(f, *args):
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic
  
two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print 'Two loop version took %f seconds' % two_loop_time 
one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print 'One loop version took %f seconds' % one_loop_time 
no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print 'No loop version took %f seconds' % no_loop_time








