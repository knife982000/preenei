import numpy as np
import matplotlib.pyplot as plt
import random


def gen_random_data(mult):
    _x = np.linspace(-1, 1, 51)
    _error = (np.random.rand(*_x.shape) - .5)
    _y = _x * mult + _error
    return _x, _y


def lineal(mult, x):
    return mult * x


def mse(y, pred):
    return np.sum(np.square(y-pred))/len(y)


def grad_mse(y, x, mult):
    return 2*np.sum((y-mult*x)*-x)/len(y)


if __name__ == '__main__':
    x, y = gen_random_data(3)
    mult = random.random() * 10 - 5
    print('Random value {}'.format(mult))
    lr = 0.01
    errors = [mse(y, lineal(mult, x))]
    print('Error: {}'.format(errors[0]))
    steps = 500
    for i in xrange(0, steps):
        mult = mult - lr * grad_mse(y, x, mult)
        e = mse(y, lineal(mult, x))
        print('{} Error: {}'.format(i, e))
        errors.append(e)
    print('Final mult: {}'.format(mult))
    plt.plot(np.asarray(range(0, steps + 1)), np.asarray(errors))
    plt.show()
    plt.plot(x, lineal(3, x), 'r-', x, lineal(mult, x), 'b-', x, y, 'g^')
    plt.show()