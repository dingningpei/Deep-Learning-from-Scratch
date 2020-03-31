

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)


        self.x = None
        self.gg = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = x
        out = np.empty((x.shape[0],self.out_channel, (x.shape[2] - (self.kernel_size - self.stride))//self.stride))
        for i in range(0, x.shape[2] - self.kernel_size + self.stride, self.stride):
            if (i+self.kernel_size > x.shape[2]):
                break
            out[:,:,i//self.stride] = np.tensordot(x[:,:,i:i + self.kernel_size], self.W,([1,2],[1,2])) +self.b
        self.out = out
        return out



    def backward(self, delta):
        dx = np.zeros((self.x.shape[0], self.x.shape[1], self.x.shape[2]))
        for b in range(delta.shape[0]):
            for j in range(self.dW.shape[0]):
                 for i in range(0, self.x.shape[2] - self.kernel_size + self.stride, self.stride):
                    if (i+self.kernel_size > self.x.shape[2]):
                        break
                    self.dW[j,:,:] += self.x[b,:,i:i+self.kernel_size] * delta[b,j,i//self.stride]
                    dx[b,:,i:i+self.kernel_size]  +=  (self.W[j,:,:].copy() * delta[b,j,i//self.stride])
        self.db = delta.sum((0,2))
        self.gg = dx
        return dx

       


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.b, self.c, self.w = x.shape
        
        return x.reshape(self.b, self.c*self.w)

    def backward(self, delta):
        
        return delta.reshape(self.b, self.c, self.w)
