import numpy as np
from activation import *

class GRU_Cell:
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()


    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        self.x = x
        self.hidden = h
        self.z = self.z_act(np.dot(self.Wzh, h.T) + np.dot(self.Wzx, x.T))
        self.r = self.r_act(np.dot(self.Wrh, h.T)+ np.dot(self.Wrx , x.T))
        self.h_tilda = self.h_act(np.dot(self.Wh , (self.r * self.hidden)) + np.dot(self.Wx,  self.x.T))
        h_t = (1 - self.z) * self.hidden + self.z * self.h_tilda

        assert self.x.shape == (self.d, )
        assert self.hidden.shape == (self.h, )

        assert self.r.shape == (self.h, )
        assert self.z.shape == (self.h, )
        assert self.h_tilda.shape == (self.h, )
        assert h_t.shape == (self.h, )
        return h_t

    def backward(self, delta):

        self.dWrx = (np.dot(delta * self.z * self.h_act.derivative(), self.Wh) * self.hidden * self.r_act.derivative()).T.dot(self.x.reshape(1,-1))
        self.dWrh = (np.dot(delta * self.z * self.h_act.derivative(), self.Wh) * self.hidden * self.r_act.derivative()).T.dot(self.hidden.reshape(1, -1))


        self.dWzx = (delta * (self.h_tilda - self.hidden) * self.z_act.derivative()).T.dot( self.x.reshape(1, -1))
        self.dWzh = (delta * (self.h_tilda - self.hidden) * self.z_act.derivative()).T.dot(self.hidden.reshape(1, -1))

        self.dWh = (delta * self.z * self.h_act.derivative()).T.dot(self.r * self.hidden.reshape(1,-1))
        self.dWx = (delta * self.z * self.h_act.derivative()).T.dot( self.x.reshape(1,-1))

        dh = delta * (1 - self.z) + np.dot(delta * (self.h_tilda - self.hidden) * self.z_act.derivative(), self.Wzh) + np.dot(delta * self.z * self.h_act.derivative(), self.Wh)* self.r + np.dot(np.dot(delta * self.z * self.h_act.derivative() , self.Wh)* self.hidden * self.r_act.derivative(), self.Wrh)
        dx = (np.dot(delta * (self.h_tilda - self.hidden) * self.z_act.derivative(), self.Wzx)) + (np.dot(delta * self.z * self.h_act.derivative(), self.Wx)) + np.dot( np.dot(delta * self.z * self.h_act.derivative(), self.Wh) * self.hidden * self.r_act.derivative(), self.Wrx)

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)



        return dx, dh
