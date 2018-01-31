import pickle as p
import numpy as np
from PIL import Image
import random
from cs231n.data_utils import load_CIFAR10
import tensorflow as tf
import cs231n.input_data as input_data


cifar10_dir='cs231n/datasets/cifar-10-batches.py/'

Xtr,Ytr,Xte,Yte=load_CIFAR10(cifar10_dir)
print ('loading data')
print ('training data shape',Xtr.shape)
print ('training labels shape',Ytr.shape)
print ('test data shape',Xte.shape)
print ('test labels shape',Yte.shape)


Xtr_rows=Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
Xte_rows=Xte.reshape(Xte.shape[0],Xte.shape[1]*Xte.shape[2]*Xte.shape[3])
colnum=Xte.shape[1]*Xte.shape[2]*Xte.shape[3]

class DataSet(object):
    def __init__(self,images,labels):
        assert images.shape[0]==labels.shape[0],("images.shape:%s labels.shape:%s"%(images.shape,labels.shape))
        self._num_examples=images.shape[0]
        assert images.shape[1]==colnum
        self._images=images
        self._labels=labels
        self._epochs_comleted=0
        self._index_in_epoch=0

    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels

    def next_batch(self,batch_size):
       start=self._index_in_epoch
       self._index_in_epoch+=batch_size
       if self._index_in_epoch > self._num_examples:
          self._epochs_completed+=1
          perm=np.arange(self._num_examples)
          np.random.shuffle(perm)
          self._images=self._images[perm]
          self._images=self._labels[perm]
          start=0
          self._index_in_epoch=batch_size
          assert batch_size<= self._num_examples
       end=self._index_in_epoch
       return self._images[start:end],self._labels[start:end]

train=DataSet(Xtr_rows,Ytr)
test=DataSet(Xte_rows,Yte)





def runs():
    x=tf.placeholder(tf.float32,[None,colnum])
    w=tf.Variable(tf.zeros([colnum,10]))
    b=tf.Variable(tf.zeros([10]))
    y=tf.nn.softmax(tf.matmul(x,w)+b)
    y_=tf.placeholder("float",[None,10])
    cross_entropy=-tf.reduce_sum(y_*tf.log(y))
    train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    init=tf.initialize_all_variables()
    sess=tf.Session()
    sess.run(init)
    for i in range(1000):
          batch_xs,batch_y=train.next_batch(100)
          sess.run(train_step,feed_dict={x:train.images,y_:train.labels})
          correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
          accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
          print (sess.run(accuracy,feed_dict={x:test.images,y_:test.labels}))

runs()
