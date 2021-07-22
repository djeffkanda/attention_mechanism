import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, drop_out=0, device='cpu'):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=n_layers, dropout=drop_out,
                            batch_first=True)

    def forward(self, inputs, hidden):
        # embed input words,
        embedded = self.embedding(inputs)
        # feed the embedding to the LSTM, then return all states
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size,device=self.device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device))
