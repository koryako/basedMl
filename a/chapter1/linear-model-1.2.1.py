'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependencies: keras 
Usage: python3 <this file>
'''

import numpy as np
import numpy.linalg as la
from keras.models import Sequential
from keras.layers import Dense, Activation

x = np.arange(-1,1,0.2)
x = np.reshape(x, [-1,1])

y = 2*x + 3

# Uncomment if you want to simulate noise in the input
noise = np.random.uniform(-0.1,0.1,y.shape)
y = y + noise

model = Sequential()
model.add(Dense(units=8, input_dim=1))
model.add(Dense(units=1))
model.summary()
model.compile(loss='mse', optimizer='sgd')
model.fit(x, y, epochs=200, batch_size=4)
ypred = model.predict(x)

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

