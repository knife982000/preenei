from keras.datasets import mnist
import scipy.misc as img
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num = x_train[0]
for i in xrange(0, len(num)):
    s = ''
    for j in xrange(0, len(num[i])):
        s += '{:03} '.format(num[i][j])
    print(s)

