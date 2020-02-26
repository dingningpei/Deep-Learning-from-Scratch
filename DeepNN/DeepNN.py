import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        if (len(hiddens) <= 0): 
            self.linear_layers = [Linear(input_size, output_size, weight_init_fn, bias_init_fn)]
        else:
            self.linear_layers =  []
            self.linear_layers.append(Linear(input_size, hiddens[0], weight_init_fn, bias_init_fn))
            for i in range(1, len(hiddens)):
                self.linear_layers.append(Linear(hiddens[i-1], hiddens[i], weight_init_fn, bias_init_fn))
            self.linear_layers.append(Linear(hiddens[-1], output_size, weight_init_fn, bias_init_fn))
        if self.bn:
            self.bn_layers = [BatchNorm(hiddens[i]) for i in range(num_bn_layers)]
        self.output = None

            


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            if i < self.num_bn_layers:
                x = self.bn_layers[i](x, (not self.train_mode))           
            x = self.activations[i](x)
        # x = self.activations[-1](x)
        self.output = x            
        return x

    def zero_grads(self):
        for i in range(len(self.linear_layers)):
            self.linear_layers[i].dW.fill(0.0)

    def step(self):
        for i in range(len(self.linear_layers)):
            self.linear_layers[i].momentum_W  = self.momentum * self.linear_layers[i].momentum_W - self.lr * self.linear_layers[i].dW
            # print(self.linear_layers[i].dW)
            self.linear_layers[i].W = self.linear_layers[i].W + self.linear_layers[i].momentum_W 
            self.linear_layers[i].momentum_b = self.momentum * self.linear_layers[i].momentum_b - self.lr * self.linear_layers[i].db
            self.linear_layers[i].b = self.linear_layers[i].b + self.linear_layers[i].momentum_b
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
                # self.bn_layers[i].gamma = self.bn_layers[i].gamma/np.sqrt(self.bn_layers[i].running_var + self.bn_layers[i].eps)
                self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta
                # self.bn_layers[i].beta = self.bn_layers[i].beta - self.bn_layers[i].gamma * self.bn_layers[i].running_mean

    def backward(self, labels):
        self.criterion.forward(self.output, labels)
        grd = self.criterion.derivative()
        # print(self.criterion.logsum)
        for i in range(self.nlayers - 1, -1,-1):
            grd = self.activations[i].derivative() * grd
            # print(grd)
            if self.bn and i < self.num_bn_layers:
                grd = self.bn_layers[i].backward(grd)
            grd = self.linear_layers[i].backward(grd)
        return grd

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    for e in range(nepochs):
        print(e)
        t_row= np.arange(trainx.shape[0])
        np.random.shuffle(t_row)
        trainx = trainx[t_row,:]
        trainy = trainy[t_row,:]
        # print(t_row == idxs)
        # Per epoch setup ...
        batchmean = []
        batchtotal = []
        for b in range(0, len(trainx), batch_size):
           mlp.zero_grads()
           mlp.forward(trainx[b:b+batch_size, :])
           mlp.backward(trainy[b:b+batch_size, :])
           batchtotal.append(mlp.total_loss(trainy[b:b+batch_size, :])/batch_size)
        #    print(type(mlp.total_loss(trainy[b:b+batch_size, :])))
           batchmean.append(mlp.error(trainy[b:b+batch_size, :])/batch_size)
           mlp.step()

        valloss = []
        valerror = []
        for b in range(0, len(valx), batch_size):
            mlp.forward(valx[b:batch_size, :])
            valloss.append(mlp.total_loss(valy[b:batch_size, :])/batch_size)
            valerror.append(mlp.error(valy[b:batch_size, :])/batch_size)
        training_errors[e] = np.array(batchmean).mean()
        training_losses[e] = np.array(batchtotal).mean()
        validation_errors[e] = np.array(valerror).mean()
        validation_losses[e] = np.array(valloss).mean()
    print(np.min(training_losses))
    print(np.min(training_errors)) 

    return (training_losses, training_errors, validation_losses, validation_errors)

