import numpy as np
from activation import *

class RNN_Cell(object):
    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = Tanh()
        h = self.hidden_size
        d = self.input_size
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        g = np.dot(x , self.W_ih.T) + self.b_ih + np.dot(h, self.W_hh.T) + self.b_hh
        h_prime = self.activation(g)
        return h_prime


    def backward(self, delta, h, h_prev_l, h_prev_t):
        batch_size = delta.shape[0]
        dz = self.activation.derivative(state=h) * delta
        self.dW_ih += np.sum(np.multiply(dz.reshape(batch_size, self.hidden_size, 1), h_prev_l.reshape(batch_size, 1, self.input_size)), axis= 0)/batch_size
        self.dW_hh += np.sum(np.multiply(dz.reshape(batch_size, self.hidden_size, 1), h_prev_t.reshape(batch_size,1,self.hidden_size)), axis = 0)/batch_size
        self.db_ih += np.sum(dz, axis = 0)/batch_size
        self.db_hh += np.sum(dz, axis = 0)/batch_size
        dx = np.dot(dz, self.W_ih)
        dh = np.dot(dz, self.W_hh)
        return dx, dh
