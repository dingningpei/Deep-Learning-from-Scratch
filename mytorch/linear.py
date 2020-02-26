import numpy as np
import math

class Linear():
    def __init__(self, in_feature, out_feature, weight_init_fn, bias_init_fn):
        self.x = None
        self.W = weight_init_fn(in_feature, out_feature)
        self.b = bias_init_fn(out_feature)
        self.dW = np.zeros((in_feature, out_feature))
        self.db = np.zeros((1, out_feature))

        self.momentum_W = np.zeros((in_feature, out_feature))
        self.momentum_b = np.zeros((1, out_feature))

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, out feature)
        """
        self.x = x
        return np.dot(x, self.W) + self.b


    def backward(self, delta):

        """
        Argument:
            delta (np.array): (batch size, out feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        for i in range(delta.shape[0]):
            self.dW += (self.x.T[:,i].reshape(-1,1) * delta[i,:])
        self.dW = (1/delta.shape[0])* self.dW
        self.db = np.mean(delta, axis=0, keepdims=True)
        return np.dot(delta, self.W.T)
