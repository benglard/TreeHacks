import theano, numpy, cPickle
import theano.tensor as T
from nn.utils import shared_zeros

theano.config.exception_verbosity='high'
theano.config.compute_test_value='raise'

class Container(object):

    def __init__(self):
        self.input = T.matrix()
        self.input.tag.test_value = numpy.random.randn(164, 255).astype(theano.config.floatX)
        self.output = self.input.copy()
        self.enc_output = None
        self.dec_output = None
        self.target = T.matrix()
        self.target.tag.test_value = numpy.random.randn(1800, 255).astype(theano.config.floatX)
        self.layers = []
        self.cost = 0.0

        self.momentum = 0.9
        self.min_clip = -15
        self.max_clip = 15
        self.lr = 0.1
        self.l2_decay = 0.001

    def add(self, layer, enc=False, dec=False, cost=False):
        if cost:
            self.cost = layer.forward(self.output, self.target)
            print 'Compiling'
            self.params()
            self.grads()
            self.updates()
        else:
            if enc:
                self.enc_output = layer.forward(self.output)
                self.output = self.enc_output
            elif dec:
                self.dec_output = layer.forward(self.output)
                self.output = self.dec_output

            print self.output.tag.test_value.shape#, self.output.tag.test_value
            self.layers.append(layer)
        
    def params(self):
        self.ps = sum([ l.params for l in self.layers ], [])
        print 'Params done'

    def grads(self):
        gs = T.grad(self.cost, self.ps)
        print 'Grads half way'
        self.gs = []
        #for g in self.gs:
        #    s = g.norm(2)
        #    self.gs.append(T.clip(g, self.min_clip, self.max_clip))
        self.gs = [ T.clip(g, self.min_clip, self.max_clip) for g in gs ]
        print 'Grads done'

    def updates(self):
        rv = []
        for param, grad in zip(self.ps, self.gs):
            delta = shared_zeros(param.get_value().shape)
            c = self.momentum * delta - self.lr * grad
            rv.append((param, param + c))
            rv.append((delta, c))
        self.updates = rv
        print 'Updates done'

    def make(self):
        """self.encode = theano.function(
            inputs=[self.input],
            outputs=self.enc_output,
            on_unused_input='ignore')
        print 'Encoding function done'

        self.decode = theano.function(
            inputs=[self.input, self.enc_output],
            outputs=self.dec_output,
            on_unused_input='ignore')
        print 'Decoding function done'

        self.train = theano.function(
            inputs=[self.input, self.enc_output, self.target],
            outputs=[self.cost, self.dec_output],
            updates=self.updates,
            on_unused_input='ignore')
        print 'Training function done'"""

        self.train = theano.function(
            inputs=[self.input, self.target],
            outputs=[self.cost, self.output],
            updates=self.updates,
            on_unused_input='ignore')

        self.generate = theano.function(
            inputs=[self.input],
            outputs=self.output,
            on_unused_input='ignore')
        print 'Generating function done'

    def save(self):
        f = file('model.save', 'wb')
        cPickle.dump(self.train, f, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(self.generate, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self):
        f = file('model.save', 'rb')
        self.train = cPickle.load(f)
        self.generate = cPickle.load(f)
        f.close()