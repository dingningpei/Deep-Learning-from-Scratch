import numpy as np
import os

class Criterion(object):
    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.logsum = None


    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y
        g = np.max(x, axis = 1, keepdims= True)
        ds = x - g 
        log = g + np.log(np.exp(ds).sum(axis = 1, keepdims = True))
        # self.logsum = x - np.log(np.exp(x - np.amax(x, axis=1))) - np.amax(x, axis=1)
        self.logsum = x - log
        self.loss = - np.sum(np.multiply(y, self.logsum), axis=1)
        return self.loss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        return np.exp(self.logits)/ np.exp(self.logits).sum(axis = 1, keepdims=True) - self.labels
