import torch.nn as nn
import torch


class RNNMonopod(nn.Module):
    def __init__(self, shape, input_size, output_size, num_envs):
        super().__init__()
        self.shape = shape
        self.num_envs = num_envs

        self.Input_rnn = nn.RNN(input_size, shape[0], num_layers = 1, batch_first = True)
        rnnLayer = []
        for i in range(len(shape)-1):
            rnnLayer.append(nn.RNN(shape[i], shape[i+1], num_layers = 1, batch_first = True))
            rnnLayer.append(nn.LeakyReLU())

        self.rnn = nn.ModuleList(rnnLayer)
        #self.lstm = nn.LSTM(input_size, shape[0], num_layers = 1, batch_first = True)
        #        
        #output Layer
        self.outputLayer = nn.Linear(shape[-1], output_size)

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    def forward(self, obs):
        h0 = torch.zeros(1, self.num_envs, self.shape[0]) #num_layers, batch, hidden_size
        c0 = torch.zeros(1, self.num_envs, self.shape[0])

        x, h_n = self.Input_rnn(obs, h0) 
        # output_size  (batch = obs.shape[0], L = num_seq_pos, hidden_size = shape[0])
        
        #x, (hn, cn) = self.lstm(obs, (h0, c0)) lstm are the same thing but with c0
        x = self.activation_fn(x)

        for rnn_layer in self.rnn:
            x, h_n1 = rnn_layer(x, h0)
            x = self.activation_fn(x)

        x = self.outputLayer(x[:,-1,:])

        return x
        
