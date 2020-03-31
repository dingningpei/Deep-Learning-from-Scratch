

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *

class CNN(object):
    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        self.convolutional_layers = []
        self.convolutional_layers.append(Conv1D(num_input_channels, num_channels[0], kernel_sizes[0], strides[0],conv_weight_init_fn, bias_init_fn))
        wide = (input_width - (kernel_sizes[0] - strides[0]))//strides[0]
        for i in range(1, len(activations)):
            self.convolutional_layers.append(Conv1D(num_channels[i-1], num_channels[i], kernel_sizes[i], strides[i], conv_weight_init_fn, bias_init_fn))
            wide = (wide - (kernel_sizes[i] -  strides[i]))//strides[i]
        self.flatten = Flatten()
        self.linear_layer = Linear( wide * num_channels[-1], num_linear_neurons, linear_weight_init_fn,bias_init_fn)

    def forward(self, x):
        for i in range(len(self.convolutional_layers)):
            x = self.convolutional_layers[i](x)
            x = self.activations[i](x)        
        x = self.flatten(x)
        x = self.linear_layer(x)
        self.output = x

        return self.output

    def backward(self, labels):
        m, _ = labels.shape
        self.loss = self.criterion(self.output, labels).sum()
        grad = self.criterion.derivative()
        grad = self.linear_layer.backward(grad)
        grad = self.flatten.backward(grad)
        for i in range(len(self.convolutional_layers)-1, -1, -1):
            grad = self.activations[i].derivative() * grad
            grad = self.convolutional_layers[i].backward(grad)
        return grad


    def zero_grads(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].dW.fill(0.0)
            self.convolutional_layers[i].db.fill(0.0)

        self.linear_layer.dW.fill(0.0)
        self.linear_layer.db.fill(0.0)

    def step(self):
        for i in range(self.nlayers):
            self.convolutional_layers[i].W = (self.convolutional_layers[i].W -
                                              self.lr * self.convolutional_layers[i].dW)
            self.convolutional_layers[i].b = (self.convolutional_layers[i].b -
                                  self.lr * self.convolutional_layers[i].db)

        self.linear_layer.W = (self.linear_layer.W - self.lr * self.linear_layers.dW)
        self.linear_layers.b = (self.linear_layers.b -  self.lr * self.linear_layers.db)


    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False
