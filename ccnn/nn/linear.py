from nn.layer import Layer
from nn.utils import shared_normal, shared_zeros
from math import sqrt
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Linear(Layer):

    def __init__(self, n_in, n_out):
        Layer.__init__(self)

        self.n_in = n_in
        self.n_out = n_out

        scale = sqrt(1.0/n_in)
        self.W = shared_normal((self.n_in, self.n_out), scale=scale)
        self.b = shared_zeros((self.n_out,))
        self.params = [ self.W, self.b ]

    def forward(self, input):
        return T.dot(input, self.W) + self.b

class Reshape(Layer):

    def __init__(self, shape):
        Layer.__init__(self)
        self.shape = shape

    def forward(self, input):
        return T.reshape(input, self.shape)

class Transfer(Layer):

    def __init__(self, pattern):
        Layer.__init__(self)
        self.pattern = pattern

    def forward(self, input):
        return input.dimshuffle(self.pattern)

class Dropout(Layer):

    def __init__(self, p=0.5):
        Layer.__init__(self)
        self.p = p
        self.training = True
        self.rng = RandomStreams()

    def forward(self, input):
        if self.training:
            return input * self.rng.binomial(
                input.shape,
                p=(1-self.p),
                dtype=theano.config.floatX)
        else:
            return input

class Identity(Layer):
    __init__ = lambda self: Layer.__init__(self)
    forward = lambda self, input: input        