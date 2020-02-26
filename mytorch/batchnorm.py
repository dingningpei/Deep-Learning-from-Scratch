import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        self.x = x
        if eval:
        #    self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
        #    self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
           xg = (self.x - self.running_mean)* (1.0/np.sqrt(self.running_var + self.eps))
           self.out = np.multiply(xg, self.gamma) + self.beta
           return self.out
        self.mean = np.mean(x, axis=0, keepdims=True)
        # self.var = np.mean(np.square(x - self.mean), axis=0, keepdims=True)
        self.var = np.var(x, axis = 0, keepdims= True)
        self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        self.out = np.multiply(self.norm, self.gamma) + self.beta
        self.running_mean = self.alpha * self.running_mean + (1.0 - self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var + (1.0 - self.alpha) * self.var

        return self.out


    def backward(self, delta):
        N, D = delta.shape
        self.dbeta = np.sum(delta, axis=0, keepdims=True) 
        d_xhat = np.multiply(delta, self.gamma)
        self.dgamma = np.sum(np.multiply(self.norm, delta), axis=0, keepdims=True)
        # d_xhat = np.multiply(delta, self.gamma)
        dvar = - 0.5 * np.sum(d_xhat * (self.x - self.mean) * ((self.var + self.eps) ** (-3.0 / 2.0)), axis=0, keepdims=True)
        dmean = - np.sum(d_xhat * ((self.var + self.eps) ** - (1.0 / 2.0)), axis=0, keepdims=True) - 2.0 / N * dvar * np.sum((self.x - self.mean), axis=0, keepdims=True)
        dx = d_xhat * ((self.var + self.eps) ** (-1.0 / 2.0)) + dvar * (2 / N) * (self.x - self.mean) + dmean * 1 / N
        return dx
