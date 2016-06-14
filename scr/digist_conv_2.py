from keras.datasets import mnist
from keras.layers import *
from keras.models import Sequential
import numpy as np

def to_output(y):
    y_r = np.zeros((y.shape[0], 10))
    for i, v in enumerate(y):
        y_r[i, v] = 1
    return y_r


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]))
    x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))
    x_train /= 255
    x_test /= 255

    model = Sequential()
    model.add(Convolution2D(nb_filter=16, nb_row=3, nb_col=3, input_shape=(x_train.shape[1],
                                                                           x_train.shape[2],
                                                                           x_train.shape[3]),
                            activation='relu'))
    model.add(MaxPooling2D())
    print model.output_shape
    model.add(Convolution2D(nb_filter=16, nb_row=3, nb_col=3,
                            activation='relu'))
    model.add(MaxPooling2D())
    print model.output_shape
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('Model score: {}'.format(model.evaluate(x_test, to_output(y_test), verbose=False)))
    raw_input('Press a key...')
    model.fit(x_train, to_output(y_train), nb_epoch=10)
    print('Model score: {}'.format(model.evaluate(x_test, to_output(y_test), verbose=False)))
