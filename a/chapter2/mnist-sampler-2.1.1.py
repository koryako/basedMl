'''
Author: Rowel Atienza
Project: https://github.com/roatienza/dl-keras
Dependency: keras 2.0
Usage: python3 <this file>
'''

import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

indexes = np.random.randint(0,x_train.shape[0], size=10)
images = x_train[indexes]
labels = y_train[indexes]
for i in range(len(indexes)):
    filename = "mnist%d.png" % labels[i]
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.savefig(filename)
    plt.show()

plt.close('all')
