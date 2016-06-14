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


def exp_error(y, x, mul_vals):
    def single_error(m):
        return mse(y, lineal(m, x))
    _s = np.vectorize(single_error)
    return _s(mul_vals)

if __name__ == '__main__':
    x, y = gen_random_data(3)
    muls = np.linspace(0, 6, 51)
    plt.plot(muls, exp_error(y, x, muls))
    plt.show()
