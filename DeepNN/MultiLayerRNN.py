import numpy as np
import sys

sys.path.append('mytorch')
from rnn_cell import *
from linear import *

class RNN_Phoneme_Classifier(object):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = [RNN_Cell(input_size, hidden_size) if i == 0 else
                    RNN_Cell(hidden_size, hidden_size) for i in range(num_layers)]
        self.output_layer = Linear(hidden_size, output_size)

        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.init_weights(*linear_weights)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0
        self.x = x
        self.hiddens.append(hidden.copy())
        for i in range(seq_len):
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
            hidden[0] = self.rnn[0].forward(x[:,i], self.hiddens[i][0])
            for j in range(1, self.num_layers):
                hidden[j] = self.rnn[j].forward(hidden[j - 1], self.hiddens[i][j])
            self.hiddens.append(hidden)
        logits = self.output_layer(self.hiddens[-1][-1])
        return logits

    def backward(self, delta):
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)
        dnext = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        for i in range(seq_len - 1, -1 , -1):
        
            for j in range(self.num_layers - 1, -1, -1):
                if ( j == 0):
                    a, b = self.rnn[j].backward(dh[j], self.hiddens[i + 1][j], self.x[:,i], self.hiddens[i][j] )
                    dh[j] = b
                else:
                    a, b = self.rnn[j].backward(dh[j], self.hiddens[i + 1][j], self.hiddens[i + 1][j-1], self.hiddens[i][j] )
                    dh[j - 1] += a
                    dh[j] = b
        return dh/batch_size
