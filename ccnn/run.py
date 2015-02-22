import nn, numpy, theano, os, binascii
from time import time

def to_data(char):
    arr = numpy.zeros((1, 255), dtype=theano.config.floatX)
    arr[0, ord(char)] = 1
    return arr

network = nn.Container()
network.add(nn.LSTM(255, 100, 100), enc=True)
network.add(nn.LSTM(100, 100, 100), enc=True)
network.add(nn.LSTM(100, 100, 100), dec=True)
network.add(nn.LSTM(100, 255, 2025), dec=True)
#network.add(nn.LSTM(100, 100, 100), dec=True)
#network.add(nn.Reshape((1, 100*100)), dec=True)
#network.add(nn.Linear(100*100, 255), dec=True)
network.add(nn.SoftMax(), dec=True)
network.add(nn.ClassNLL(), cost=True)
print 'Network created'
print 'Compiling functions'
network.make()
print 'Functions created'

path = '../data/'
n_train = 900
for n in xrange(1000):
    cname = os.path.join(path, '{}.c'.format(n))
    oname = os.path.join(path, '{}.o'.format(n))

    with open(cname, 'r') as fi:
        code = fi.read()
    with open(oname, 'rb') as fi:
        bin = binascii.hexlify(fi.read())

    code = '{}{}'.format(code, chr(3))
    bin = '{}{}'.format(bin, chr(3))

    #print code, bin
    print len(code), len(bin)

    x = numpy.zeros((len(code), 255), dtype=theano.config.floatX)
    for k, elem in enumerate(code):
        x[k] = to_data(elem).reshape(255)
        #enco = network.encode(xd)

    y = numpy.zeros((len(bin), 255), dtype=theano.config.floatX)
    for k, elem in enumerate(bin):
            y[k] = to_data(elem).reshape(255)


    if n < n_train:
        s = time()
        cost, output = network.train(x, y)
        print n, time() - s, cost
    else:
        output = network.generate(x)

    outtext = ''
    for row in output:
        am = row.argmax()
        if am == 3:
            break
        else:
            outtext += chr(am)
    print len(outtext), '' if n < n_train else outtext
    print '-------'

"""
x = numpy.random.randn(1, 255).astype(theano.config.floatX)
y = numpy.random.randn(1, 255).astype(theano.config.floatX)
enco = network.encode(x)
print network.train(x, enco, y)
"""