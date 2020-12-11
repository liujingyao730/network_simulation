import torch
import torch.nn as nn
import numpy as np

class gcn(nn.Module):

    def __init__(self, input_size, output_size, activation="relu"):

        super(gcn, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activation == nn.Sigmoid()
        else:
            raise NotImplementedError

        self.con_kernal = nn.parameter.Parameter(torch.Tensor(self.input_size, self.output_size))

        torch.nn.init.xavier_normal_(self.con_kernal)

    def forward(self, input_data, laplace_list):

        output = torch.einsum("cc,bci->bci", laplace_list, input_data)
        output = torch.einsum("bci,io->bco", output, self.con_kernal)
        output = self.activation(output)

        return output

class gat(nn.Module):

    def __init__(self, input_size, output_size):

        super(gat, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.W = nn.Parameter(torch.zeros(size=(self.input_size, self.output_size)))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_size, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.relu = nn.ReLU()
    
    def forward(self, inputs, adj):

        h = torch.einsum("bni,io->bno", inputs, self.W)
        batch_size = h.size()[0]
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(batch_size, N*N, -1), h.repeat(1, N, 1)], dim=2).view(batch_size, N, -1, 2*self.output_size)
        e = self.relu(torch.einsum("bmnc,cd->bmnd", a_input, self.a).squeeze(3))
        
        adj = adj.repeat(batch_size, 1, 1)
        zero_vector = torch.zeros_like(adj)

        attention = torch.where(adj > 0, e, zero_vector)
        attention = torch.nn.functional.softmax(attention, dim=2)
        h_prime = torch.einsum("bmn,bno->bmo", attention, h)

        return h_prime

if __name__ == "__main__":
    g = gat(8, 16)
    input_data = torch.rand(10, 40, 8).float()
    laplace = torch.rand(40, 40).float()
    output = g(input_data, laplace)
