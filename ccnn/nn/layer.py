class Layer(object):

    def __init__(self):
        self.params = []
        self.cost = 0

    def forward(self):
        raise NotImplementedError("Layer.forward has no implementation")