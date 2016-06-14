from keras.datasets import mnist
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt


def to_output(y):
    y_r = np.zeros((y.shape[0], 10))
    for i, v in enumerate(y):
        y_r[i, v] = 1
    return y_r


def calc_accuracy():
    p = predict(x_test)
    p = np.argmax(p, axis=-1)
    acc = float(np.sum(p == y_test)) / float(y_test.shape[0])
    print('Accuracy: {}'.format(acc))


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] ** 2))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] ** 2))
    x_train /= 255
    x_test /= 255

    X = T.matrix()
    Y = T.matrix()
    w = theano.shared(np.zeros((28 ** 2, 10), dtype=theano.config.floatX))
    cost = T.mean(T.sqr(Y - T.dot(X, w)))
    gradient = T.grad(cost=cost, wrt=w)
    updates = [[w, w - gradient * 0.1]]
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    predict = theano.function(inputs=[X], outputs=T.dot(X, w))
    calc_accuracy()
    raw_input('Press enter')
    errors = []
    steps = 40
    for i in xrange(0, steps):
        e = train(x_train, to_output(y_train))
        print('{} Error: {}'.format(i, e))
        errors.append(e)
    print('Final mult: {}'.format(w.get_value()))
    plt.plot(np.asarray(range(0, steps)), np.asarray(errors))
    plt.show()
    calc_accuracy()
