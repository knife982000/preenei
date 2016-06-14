from keras.datasets import mnist
import scipy.misc as img
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
for i in xrange(0, 10):
    img.imsave('num_{}.jpg'.format(i), x_train[i])
