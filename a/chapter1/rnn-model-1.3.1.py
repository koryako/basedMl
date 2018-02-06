'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependencies: keras 
Usage: python3 <this file>
'''

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.optimizers import Adam

timesteps = 30
input_dim = 12
hidden_units = 512
n_activities = 5
model = Sequential()
model.add(SimpleRNN(units=hidden_units, dropout=0.2, \
        input_shape=(timesteps, input_dim)))
model.add(Dense(n_activities, activation='softmax'))
model.compile(loss='categorical_crossentropy', \
        optimizer=Adam(), metrics=['accuracy'])
model.summary()
