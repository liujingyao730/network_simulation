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

        output = torch.einsum("nm,bmi->bni", laplace_list, input_data)
        output = torch.einsum("bni,io->bno", output, self.con_kernal)
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
    
    def forward(self, inputs, index, weight=None):

        h = torch.einsum("bni,io->bno", inputs, self.W)
        batch_size = h.size()[0]
        N = h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, 2).view(batch_size, N, 2, -1), h[:, index, :]], dim=2).view(batch_size, N, -1, 2*self.output_size)
        e = self.relu(torch.einsum("bmnc,cd->bmnd", a_input, self.a).squeeze(3))
        
        weight = weight.unsqueeze(0).repeat(batch_size, 1, 1)
        zero_vector = torch.zeros_like(weight)

        attention = torch.where(weight > 0, e, zero_vector)
        attention = torch.nn.functional.softmax(attention, dim=2)
        attention = attention.unsqueeze(2)
        h_prime = torch.einsum("bcmn,bcno->bcmo", attention, h[:, index, :])
        h_prime = h_prime.squeeze(2)

        return h_prime
    
class dcn(nn.Module):

    def __init__(self, input_size, output_size, activation="relu"):

        super(dcn, self).__init__()

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

        self.out_weight = nn.parameter.Parameter(torch.Tensor(self.input_size, self.output_size))
        self.in_weight = nn.parameter.Parameter(torch.Tensor(self.input_size, self.output_size))

        torch.nn.init.xavier_normal_(self.out_weight)
        torch.nn.init.xavier_normal_(self.in_weight)
    
    def forward(self, input_data, laplace):

        out_h = torch.einsum("cc,bci->bci", laplace, input_data)
        out_h = torch.einsum("bci,io->bco", out_h, self.out_weight)
        
        in_h = torch.einsum("cc,bci->bci", laplace.transpose(0, 1), input_data)
        in_h = torch.einsum("bci,io->bco", in_h, self.in_weight)
        
        output = in_h + out_h
        output = self.activation(output)

        return output

if __name__ == "__main__":
    g = dcn(8, 16)
    input_data = torch.rand(10, 40, 8).float()
    laplace = torch.rand(40, 40).float()
    output = g(input_data, laplace)
