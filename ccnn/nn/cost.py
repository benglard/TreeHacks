from nn.layer import Layer
import theano.tensor as T
import numpy

class CostLayer(Layer):

    def __init__(self, f):
        Layer.__init__(self)
        self.f = f

    def forward(self, input, target):
        self.input = input
        return self.f(input, target)

MSE = lambda: CostLayer(lambda x, t: T.mean((x - t) ** 2))

def nll(x, t):
    shape = numpy.prod(x.tag.test_value.shape)
    nx = x.reshape((shape,))
    cost = T.log(nx)
    return -cost[T.argmax(t)]
ClassNLL = lambda: CostLayer(nll)

CrossEntropy = lambda: CostLayer(lambda x, t: T.nnet.binary_crossentropy(x, t).mean())