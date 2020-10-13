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

if __name__ == "__main__":
    g = gcn(8, 16)
    input_data = torch.rand(10, 40, 8)
    laplace = torch.rand(40, 40)
    output = g(input_data, laplace)
