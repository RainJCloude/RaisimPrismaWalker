import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, num_envs):
        super().__init__()
        self.shape = shape

        self.rnn = nn.RNN(input_size, shape[0], num_layers = 1, batch_first = True)
        self.rnnLayer = []
        for i in range(len(shape)-1):
            self.rnnLayer.append(nn.RNN(shape[i], shape[i+1], num_layers = 1, batch_first = True))

        #self.lstm = nn.LSTM(input_size, shape[0], num_layers = 1, batch_first = True)
        self.activation_fn = self.activation_fn()
        
        #output Layer
        self.outputLayer = nn.Linear(shape[-1], output_size)


    def forward(self, obs, num_seq_pos):
        h0 = torch.zeros(1, num_envs, self.shape[0]) #num_layers, batch, hidden_size
        c0 = torch.zeros(1, num_envs, self.shape[0])

        obs = obs.reshape(obs.shape[0], num_seq_pos, -1)
        x, h_n = self.rnn(obs, h0) 
        # output_size  (batch = obs.shape[0], L = num_seq_pos, hidden_size = shape[0])
        
        #x, (hn, cn) = self.lstm(obs, (h0, c0)) lstm are the same thing but with c0
        x = self.activation_fn(x)

        for i in range(len(self.shape)-1):
            x, h_n1 = self.rnnLayer[i](x, h0)
            x = self.activation_fn(x)

        x = self.outputLayer(x[:,-1,:])

        return x
        
