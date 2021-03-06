# -*- coding: utf-8 -*-
import tensorflow as tf 
"""
a = tf.constant(5) 
b = tf.constant(2)
c = tf.constant(3)
d = tf.multiply(a,b) 
e = tf.add(c,b) 
f = tf.subtract(e,d) 
sess = tf.Session() 
outs = sess.run(f) 
sess.close() 
print("outs = {}".format(outs))


print(tf.get_default_graph())

g = tf.Graph()
print(g)

a = tf.constant(5) 

print(a.graph is g)
print(a.graph is tf.get_default_graph())


g1 = tf.get_default_graph() 
g2 = tf.Graph() 

print(g1 is tf.get_default_graph())

with g2.as_default(): 
    print(g1 is tf.get_default_graph())

print(g1 is tf.get_default_graph())

with tf.Session() as sess:
   fetches = [a,b,c,d,e,f]
   outs = sess.run(fetches) 

print("outs = {}".format(outs))
print(type(outs[0]))


c = tf.constant(4.0)
print(c)

c = tf.constant(4.0, dtype=tf.float64)
print(c)
print(c.dtype)


x = tf.constant([1,2,3],name='x',dtype=tf.float32) 
print(x.dtype)
x = tf.cast(x,tf.int64)
print(x.dtype)

import numpy as np 

c = tf.constant([[1,2,3],
                 [4,5,6]]) 
print("Python List input: {}".format(c.get_shape()))

c = tf.constant(np.array([
                 [[1,2,3], 
                  [4,5,6]], 

                 [[1,1,1], 
                  [2,2,2]]
                 ])) 

print("3d Numpy array input: {}".format(c.get_shape()))
"""




def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def Noramal_dis(dim,mean = 0,std = 1,Truncated=False):
    # === Noramal and Truncated normal distributions ===截断正态分布
    if Truncated is True:
        x = tf.truncated_normal((1,dim),mean,std).eval()
        print("truncated = {}".format(x),"shape = {}".format(x.shape))
    else:
        x = tf.random_normal((1,dim),mean,std).eval()
        print("Noramal = {}".format(x),"shape = {}".format(x.shape))
    return x

def Uniform_dist(dim,minval = -2,maxval = 2):
    # === Uniform distribution均匀分布
    x_uniform = tf.random_uniform((1,dim),minval,maxval).eval()
    print("uniform = {}".format(x_uniform),"shape = {}".format(x_uniform.shape))
    return x_uniform

def distributions(dim):
    x_uniform=Uniform_dist(dim)
    x_truncated=Noramal_dis(dim,Truncated=True)
    x_normal=Noramal_dis(dim,Truncated=False)
    return x_normal,x_truncated,x_uniform

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
#     ax.set_ylim([-1.1,1.1])
    ax.tick_params(axis='both', which='major', labelsize=15)
    
def get_axis_limits(ax, scale=.8):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale

def distributions_show():
    import matplotlib.pyplot as plt 

    f,axarr = plt.subplots(1,3,figsize=[15,4],sharey=True)
    titles = ['Normal','Truncated Normal','Uniform']

    print(x_normal.shape)
    for i,x in enumerate([x_normal,x_truncated,x_uniform]):
        ax = axarr[i]
        ax.hist(x[0],bins=100,color='b',alpha=0.4)
        ax.set_title(titles[i],fontsize=20)
        ax.set_xlabel('Values',fontsize=20)
        ax.set_xlim([-5,5])
        ax.set_ylim([0,1800])
    
        simpleaxis(ax)
    
        axarr[0].set_ylabel('Frequency',fontsize=20)
        plt.suptitle('Initialized values',fontsize=30, y=1.15)


    for ax,letter in zip(axarr,['A','B','C']):
        simpleaxis(ax)
        ax.annotate(letter, xy=get_axis_limits(ax),fontsize=35)

    plt.tight_layout()
    plt.savefig('histograms.png', bbox_inches='tight', format='png', dpi=200, pad_inches=0,transparent=True)
    plt.show()

if __name__ == '__main__':
    #sess = tf.InteractiveSession()
    #x_normal,x_truncated,x_uniform=distributions(50000)
    #sess.close()
    #distributions_show()

    # As usual, a bit of setup
   # from __future__ import print_function
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    #from cs231n.classifiers.fc_net import *
    #from cs231n.data_utils import get_CIFAR10_data
    #from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
    #from cs231n.solver import Solver


    #plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    #plt.rcParams['image.interpolation'] = 'nearest'
    #plt.rcParams['image.cmap'] = 'gray'

    # for auto-reloading external modules
    # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython



