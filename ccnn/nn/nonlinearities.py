from nn.layer import Layer
import theano.tensor as T

class NonLinear(Layer):

    def __init__(self, f):
        Layer.__init__(self)
        self.f = f

    def forward(self, input):
        return self.f(input)

Tanh = lambda: NonLinear(T.tanh)
Sigmoid = lambda: NonLinear(T.nnet.sigmoid)
ReLU = lambda: NonLinear(lambda x: ((x + abs(x)) / 2.0))

def sm(x):
    e_x = T.exp(x - x.max(axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)
SoftMax = lambda: NonLinear(sm) #T.nnet.softmax