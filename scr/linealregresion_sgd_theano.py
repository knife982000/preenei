import numpy as np
import matplotlib.pyplot as plt
import theano
from theano import tensor as T

def gen_random_data(mult):
    _x = np.linspace(-1, 1, 51)
    _error = (np.random.rand(*_x.shape) - .5)
    _y = _x * mult + _error
    return _x, _y


if __name__ == '__main__':
    x, y = gen_random_data(3)
    X = T.vector()
    Y = T.vector()
    w = theano.shared(np.asarray(0., dtype=theano.config.floatX))
    cost = T.mean(T.sqr(Y - w * X))
    gradient = T.grad(cost=cost, wrt=w)
    updates = [[w, w - gradient * 0.01]]
    train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    errors = []
    steps = 500
    for i in xrange(0, steps):
        e = train(x, y)
        print('{} Error: {}'.format(i, e))
        errors.append(e)
    print('Final mult: {}'.format(w.get_value()))
    plt.plot(np.asarray(range(0, steps)), np.asarray(errors))
    plt.show()
    plt.plot(x, 3 * x, 'r-', x, w.get_value() * x, 'b-', x, y, 'g^')
    plt.show()