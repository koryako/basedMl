import cv2
import glob
import time
import pickle
import numpy as numpy
import matplotlic.image as mpimg
import matplotlib.pyplot as pyplot
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.measurements import label

