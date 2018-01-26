# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
from util.data_utils import get_CIFAR10_data


data = get_CIFAR10_data('../datasets/cifar-10-batches-py')

X_train=data['X_train']
y_train=data['y_train']
X_val=data['X_val']
y_val=data['y_val']
X_test=data['X_test']
y_test=data['y_test']
X_train=np.transpose(X_train,(0,2,3,1))
X_val=np.transpose(X_val,(0,2,3,1))
X_test=np.transpose(X_test,(0,2,3,1))
print('Train data shape:{} '.format(X_train.shape))
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# clear old variables
tf.reset_default_graph()

# setup input (e.g. the data that changes every batch)
# The first dim is None, and gets sets automatically based on batch size fed in
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
# define model
def complex_model(X,y,is_training):
    a1=tf.nn.conv2d(X, [7,7,3,32], strides=[1,1,1,1], padding='same')
    a1 = tf.nn.relu(a1)
    bn1=tf.nn.batch_normalization(a1)
    pass


def simple_model(X,y):
    # define our weights (e.g. init_two_layer_convnet)
    
    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    W1 = tf.get_variable("W1", shape=[5408, 10])
    b1 = tf.get_variable("b1", shape=[10])

    # define our graph (e.g. two_layer_convnet)
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    h1_flat = tf.reshape(h1,[-1,5408])
    y_out = tf.matmul(h1_flat,W1) + b1
    return y_out

#y_out = simple_model(X,y)
y_out = complex_model(X,y,is_training)

"""
# define our loss
total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         y: yd[idx],
                         is_training: training_now }
            # get batch size
            actual_batch_size = yd[idx].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct

with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        sess.run(tf.global_variables_initializer())
        print('Training')
        run_model(sess,y_out,mean_loss,X_train,y_train,1,64,100,train_step,True)
        print('Validation')
        run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
"""

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

# Now we're going to feed a random batch into the model 
# and make sure the output is the right size
x = np.random.randn(64, 32, 32,3)
with tf.Session() as sess:
    with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0"
        tf.global_variables_initializer().run()

        ans = sess.run(y_out,feed_dict={X:x,is_training:True})
        
        print(ans.shape)
        print(np.array_equal(ans.shape, np.array([64, 10])))
"""
m = K.mean(X, axis=-1, keepdims=True)#计算均值  
std = K.std(X, axis=-1, keepdims=True)#计算标准差  
X_normed = (X - m) / (std + self.epsilon)#归一化  
out = self.gamma * X_normed + self.beta#重构变换  
"""


"""
1、《Batch Normalization: Accelerating Deep Network Training by  Reducing Internal Covariate Shift》

2、《Spatial Transformer Networks》

3、https://github.com/fchollet/keras
"""

"""
input_shape = self.input_shape  
 reduction_axes = list(range(len(input_shape)))  
 del reduction_axes[self.axis]  
 broadcast_shape = [1] * len(input_shape)  
 broadcast_shape[self.axis] = input_shape[self.axis]  
 if train:  
     m = K.mean(X, axis=reduction_axes)  
     brodcast_m = K.reshape(m, broadcast_shape)  
     std = K.mean(K.square(X - brodcast_m) + self.epsilon, axis=reduction_axes)  
     std = K.sqrt(std)  
     brodcast_std = K.reshape(std, broadcast_shape)  
     mean_update = self.momentum * self.running_mean + (1-self.momentum) * m  
     std_update = self.momentum * self.running_std + (1-self.momentum) * std  
     self.updates = [(self.running_mean, mean_update),  
                     (self.running_std, std_update)]  
     X_normed = (X - brodcast_m) / (brodcast_std + self.epsilon)  
 else:  
     brodcast_m = K.reshape(self.running_mean, broadcast_shape)  
     brodcast_std = K.reshape(self.running_std, broadcast_shape)  
     X_normed = ((X - brodcast_m) /  
                 (brodcast_std + self.epsilon))  
 out = K.reshape(self.gamma, broadcast_shape) * X_normed + K.reshape(self.beta, broadcast_shape)  
 """

"""
def bn(x, is_training):  
    x_shape = x.get_shape()  
    params_shape = x_shape[-1:]  
  
    axis = list(range(len(x_shape) - 1))  
  
    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer())  
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer())  
  
    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer(), trainable=False)  
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer(), trainable=False)  
  
    # These ops will only be preformed when training.  
    mean, variance = tf.nn.moments(x, axis)  
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)  
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)  
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)  
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)  
  
    mean, variance = control_flow_ops.cond(  
        is_training, lambda: (mean, variance),  
        lambda: (moving_mean, moving_variance))  
  
    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)  
"""

"""
另外，这里有使用batch
 normalization的示例：martin-gorner/tensorflow-mnist-tutorial
 https://github.com/martin-gorner/tensorflow-mnist-tutorial/blob/master/mnist_4.2_batchnorm_convolutional.py

还可以参考：resnet：https://github.com/MachineLP/tensorflow-resnet

还可以看大师之作：CNN和RNN中如何引入BatchNorm   http://blog.csdn.net/malefactor/article/details/51549771

训练好的模型加载：tensorflow中batch normalization的用法  http://www.cnblogs.com/hrlnw/p/7227447.html
"""