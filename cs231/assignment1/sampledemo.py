
import pickle as p
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as plimg
from PIL import Image
import random
from data_utils import load_CIFAR10
import tensorflow as tf
# Import MINST data
import cs231n.input_data as input_data

#mnist = input_data.read_data_sets("data_i", one_hot=True)
cifar10_dir='../../../../datasets/cifar-10-batches.py/'
#cifar10_dir = 'cs231n/datasets/cifar-10-batches.py/'
Xtr, Ytr, Xte, Yte = load_CIFAR10(cifar10_dir) # a magic function we provide
print "loading data"
print 'Training data shape: ', Xtr.shape 
print 'Training labels shape: ', Ytr.shape
print 'Test data shape: ', Ytr.shape
print 'Test labels shape: ', Yte.shape
# flatten out all images to be one-dimensional
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
Xte_rows = Xte.reshape(Xte.shape[0],32 *32 *3) # Xte_rows becomes 10000
print Xte_rows
#colnum=Xtr_rows.shape[1]*Xtr_rows.shape[2]*Xtr_rows.shape[3]

#images=Xtr_rows.reshape(Xtr_rows.shape[0],colnum)
#labels=Xte_rows.reshape(Xte_rows.shape[0],colnum)
"""
class DataSet(object):
    def __init__(self,images,labels):
	   assert images.shape[0]==labels.shape[0],("images.shape:%s labels.shape:%s" %(images.shape,labels.shape))
       self._num_examles=images.shape[0]
	   assert images.shape[3]==1
	   self._images=images
	   self._labels=labels
	   self._epochs_completed=0
	   self._index_in_epoch=0
	   
	@property
	def imges(self):
	   return self._images
	@property
	def labels(self):
	   return self._labels
	  
	def next_batch(self,batch_size):
	   start=self._index_in_epoch
	   self._index_in_epoch+=batch_size
	   if self._index_in_epoch> self._num_examples:
	      self._epochs_completed +=1
		  perm=np.arange(self._num_examples)
		  np.random.shuffle(perm)
		  self._images=self._images[perm]
		  self._images=self._labels[perm]
		  start=0
		  self._index_in_epoch=batch_size
		  assert batch_size<=self._num_examples
	    end=self.index_in_epoch
	    return self._images[start:end],self._labels[start:end]
		
train=DataSet(Xtr_rows,Ytr)		
test=DataSet(Xte_rows,Yte)		
"""	 
"""  
def runs():
      x = tf.placeholder(tf.float32, [None, colnum])
      W = tf.Variable(tf.zeros([colnum,10]))
      b = tf.Variable(tf.zeros([10]))
      y = tf.nn.softmax(tf.matmul(x,W) + b)
      y_ = tf.placeholder("float", [None,10])#训练变量
      cross_entropy = -tf.reduce_sum(y_*tf.log(y))
      train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
      init = tf.initialize_all_variables()
      sess = tf.Session()
      sess.run(init)
      for i in range(1000):
             batch_xs, batch_ys = train.next_batch(100) #获取100张图片数据是一个二维数据【100，784】 和 标签二维数组 【100，10】作为训练用数据
             sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
             correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
             accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
             print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



def load_CIFAR_batch(filename):
    # load single batch of cifar 
    with open(filename, 'rb')as f:
        datadict = p.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y

def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        lines = [x for x in f.readlines()]
        print(lines)

def save_image():
    load_CIFAR_Labels("batches.meta")
    imgX, imgY = load_CIFAR
    batch("data_batch_1")
    print imgX.shape
    print "saving:"
    for i in xrange(imgX.shape[0]):
        imgs = imgX[i - 1]
        if i < 100:
            img0 = imgs[0]
            img1 = imgs[1]
            img2 = imgs[2]
            i0 = Image.fromarray(img0)
            i1 = Image.fromarray(img1)
            i2 = Image.fromarray(img2)
            img = Image.merge("RGB",(i0,i1,i2))
            name = "img" + str(i)
            img.save("data/images/"+name,"png")
            for j in xrange(imgs.shape[0]):
                img = imgs[j - 1]
                name = "img" + str(i) + str(j) + ".png"
                print "save" + name
     print ("saved finish.")

"""
#if __name__ == "__main__":
    #save_image()