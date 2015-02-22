import nn, numpy, theano, os, binascii
from time import time

def to_data(char):
    arr = numpy.zeros((1, 255), dtype=theano.config.floatX)
    arr[0, ord(char)] = 1
    return arr

network = nn.Container()
network.add(nn.LSTM(255, 100, 100), enc=True)
network.add(nn.LSTM(100, 255, 1), dec=True)
network.add(nn.SoftMax(), dec=True)
network.add(nn.ClassNLL(), cost=True)
print 'Network created'
print 'Compiling functions'
network.make()
print 'Functions created'

path = '../data/'
n_train = 900
for n in xrange(n_train):
    with open(os.path.join(path, '{}.c'.format(n)), 'r') as fi:
        code = '{}{}'.format(fi.read(), chr(3))
    print n, code, len(code)

    for c1, c2 in zip(code[:-1], code[1:]):
        x = to_data(c1)
        y = to_data(c2)
        s = time()
        cost, output = network.train(x, y)
        print c1, c2, time() - s, cost, output.argmax(), chr(output.argmax())
