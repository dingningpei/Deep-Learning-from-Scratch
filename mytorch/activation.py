import numpy as np
import os


class Activation(object):


    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):


    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state = 1.0 / (1.0 + np.exp(-x))
        return self.state

    def derivative(self):
        return np.multiply(self.state, 1 - self.state)


class Tanh(Activation):

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.state

    def derivative(self):
        return 1 - np.square(self.state)


class ReLU(Activation):

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.maximum(x, 0)
        return self.state

    def derivative(self):
        return np.where(self.state <= 0, self.state, 1)
