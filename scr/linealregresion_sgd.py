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


def grad_mse(y, x, mult):
    return 2*np.sum((y-mult*x)*-x)/len(y)


def grad_checking(y, x, mult, epsilon=1e-6 ):
    ge = (mse(y, lineal(mult+epsilon, x)) - mse(y, lineal(mult-epsilon, x))) / (2 * epsilon)
    ga = grad_mse(y, x, mult)
    print '{} Gradiente estimado {}, analitico {}, diferencia {}, val: {}'.format(mult, ge, ga,
                                                                                  abs(ge - ga),
                                                                                  abs(ge - ga) < epsilon)

if __name__ == '__main__':
    x, y = gen_random_data(3)
    for i in xrange(0, 100):
        grad_checking(y, x, i)
