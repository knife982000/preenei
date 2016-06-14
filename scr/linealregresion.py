import numpy as np
import matplotlib.pyplot as plt


def gen_random_data(mult):
    _x = np.linspace(-1, 1, 51)
    _error = (np.random.rand(*_x.shape) - .5)
    _y = _x * mult + _error
    return _x, _y


def lineal(mult, x):
    return mult * x


def mse(y, pred):
    return np.sum(np.square(y-pred))/len(y)


if __name__ == '__main__':
    x, y = gen_random_data(3)
    plt.plot(x, y, 'ro')
    plt.show()
    plt.plot(x, y, 'ro', x, lineal(3, x))
    print('error {}'.format(mse(y, lineal(3, x))))
    plt.show()
