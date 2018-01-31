# -*- coding: utf-8 -*-
from util import load_cvs_manuel_earthquake,Normalize,val_test
import numpy as np
np.random.seed(1)
# Import data
data_total=load_cvs_manuel_earthquake()

data_total=Normalize(data_total)#
np.random.shuffle(data_total)#随机打乱

data_train_x,data_train_y,val_x,val_y=val_test(data_total,0.8)
print (data_total.shape)


# Sigmoid function
def sigmoid(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

# Tanh function
def tanh(x,deriv=False):
    if(deriv==True):
        return 1.0 - np.tanh(x)**2

    return np.tanh(x)

def forward():
    
# Learning rate

for lr in (0.1, 0.2, 0.5, 1.0):
    for hidden_layers in (6,12, 24):
        lr_cur = lr
        # Randomly initialize our weights with mean 0
        syn0 = 2*np.random.random((4,hidden_layers)) - 1
        syn1 = 2*np.random.random((hidden_layers,1)) - 1
        final_error = 0.0
        prev_error = 0.0
        # If we have a larger network, lets train it for longer
        for j in xrange(150*hidden_layers):
            for batch_idx in range(len(data_train_x)/32-1):
                input_data = data_train_x[batch_idx*32:(batch_idx+1)*32]
                output_data = data_train_y[batch_idx*32:(batch_idx+1)*32]

            	# Feed forward through layers 0, 1, and 2
                l0 = input_data
                l1 = sigmoid(np.dot(l0,syn0))
                l2 = sigmoid(np.dot(l1,syn1))

                #print l2

                # How much did we miss the target value?
                l2_error = output_data - l2

                prev_error = final_error
                final_error = np.mean(np.abs(l2_error))

                if prev_error < final_error - 0.01 and lr_cur >= 0.001:
                    lr_cur = lr_cur / 2.0

                #if (batch_idx % 10000) == 0:
                #    print "Error:" + str(np.mean(np.abs(l2_error)))
                    
                # In what direction is the target value?
                # Were we really sure? if so, don't change too much.
                l2_delta = l2_error*sigmoid(l2,deriv=True)

                l1_error = l2_delta.dot(syn1.T)
                l1_delta = l1_error * sigmoid(l1,deriv=True)

                syn1 += l1.T.dot(l2_delta) * lr_cur
                syn0 += l0.T.dot(l1_delta) * lr_cur

        print "Final Learning Rate: ", lr_cur, ", Hidden Layers: ", hidden_layers, ", Error: ", final_error
