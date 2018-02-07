# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as la
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation,SimpleRNN,Dropout
from keras.optimizers import RMSprop,Adam
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def preminist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_labels = np.amax(y_train)+1
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1,image_size*image_size])
    x_train = x_train.astype('float32')/255
    x_test = np.reshape(x_test, [-1,image_size*image_size])
    x_test = x_test.astype('float32')/255
    return x_train,y_train,x_test,y_test

def minstPlot():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    indexes = np.random.randint(0,x_train.shape[0], size=10)
    images = x_train[indexes]
    labels = y_train[indexes]
    for i in range(len(indexes)):
        filename = "mnist%d.png" % labels[i]
        image = images[i]
        plt.imshow(image, cmap='gray')
        #plt.savefig(filename)
        plt.show()

    plt.close('all')

def sampleCnn(n_digits=10):
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', \
            strides=2, input_shape=(28, 28, 1), padding='same'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu', \
            strides=2))
    model.add(Flatten())
    model.add(Dense(n_digits, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), \
             metrics=['accuracy'])
    model.summary()
    return model

    

def sampleRnn(input=[30,12,512,5]):
    timesteps = input[0]
    input_dim = input[1]
    hidden_units = input[2]
    n_activities = input[3]
    model = Sequential()
    model.add(SimpleRNN(units=hidden_units, dropout=0.2, \
        input_shape=(timesteps, input_dim)))
    model.add(Dense(n_activities, activation='softmax'))
    model.compile(loss='categorical_crossentropy', \
        optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model

def sampleData():
    x = np.arange(-1,1,0.2)
    x = np.reshape(x, [-1,1])
    y = 2*x + 3
    # Uncomment if you want to simulate noise in the input
    noise = np.random.uniform(-0.1,0.1,y.shape)
    y = y + noise
    return x,y

def fullLayers():
    model = Sequential()
    model.add(Dense(units=8, input_dim=1))
    model.add(Dense(units=1))
    model.summary()
    return model

def fullconnect(image_size,hidden_units = 256,batch_size = 128,dropout = 0.45,num_labels=10):
    # from keras.regularizers import l2
    # model.add(Dense(hidden_units, kernel_regularizer=l2(0.001), input_dim=input_size))
    input_size = image_size*image_size
    model = Sequential()
    model.add(Dense(hidden_units, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(num_labels))
    model.add(Activation('softmax'))
    model.summary()
    return model

  
def train(model,x,y,epochs=200,batch_size=4):
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss='mse', optimizer='sgd')
    model.fit(x, y, epochs=200, batch_size=4)
    #model.fit(x_train, y_train, epochs=200, batch_size=batch_size)       
    return model




def val(model,x_test,y_test,batch_size=128):
    score = np.asarray(model.evaluate(x_test, y_test, batch_size=batch_size))*100.0
    print("\nTest accuracy: %.1f%%" % score[1])
    

def test(model,x):
    ypred = model.predict(x)
    return ypred

if __name__ == '__main__':
    """
    model=fullconnect()
    x,y=sampleData()
    train(model,x,y)
    ypred=test(model,x)
    print(ypred)
    """
    #model=sampleRnn()
    #minstPlot()






























"""
ones = np.ones(x.shape)
A = np.concatenate([x,ones], axis=1)
k = np.matmul(la.pinv(A),y) 
yla = np.matmul(A,k)
outputs = np.concatenate([y, yla, ypred], axis=1)
print("Ground Truth, Linear Alg Prediction, MLP Prediction")
print(outputs)

# Uncomment to see the output for a new input data that is not part of the training data.
# x = np.array([2])
# ypred = model.predict(x)
# print(ypred)





"""
