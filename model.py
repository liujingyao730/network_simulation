import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import gcn


class test_model(nn.Module):
    def __init__(self, args):

        super().__init__()
        self.init_length = args["init_length"]
        self.output_size = args["dest_number"]

    def forward(self, input_data, adj):

        return input_data[:, self.init_length:-1, :, :]

    def infer(self, input_data, cell_list, adj):

        return input_data[:, :, :self.output_size]


class GCN_GRU(nn.Module):
    def __init__(self, args, input_cells=None):

        super(GCN_GRU, self).__init__()
        self.input_size = args["input_size"]
        self.output_size = args["output_size"]
        self.hidden_size = args.get("hidden_size", 64)
        self.init_length = args.get("init_length", 4)
        self.input_cells = input_cells

        self.gnn_type = args.get("gnn", "gcn")
        if self.gnn_type == "gcn":
            self.init_graph = gcn(self.input_size, self.hidden_size)
            self.forward_gnn = gcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = gcn(self.hidden_size, self.hidden_size)
        else:
            raise NotImplementedError
        self.sptial_merge = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.rnn_type = args.get("rnn", "gru")
        if self.rnn_type == "gru":
            self.temporal_cell = nn.GRUCell(self.input_size, self.hidden_size)

        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def adj_to_laplace(self, adj_list):

        D_tilde = torch.diag_embed(
            torch.pow(torch.sum(adj_list, dim=1), -1 / 2))
        laplace = torch.einsum("tbc,tcd->tbd", D_tilde, adj_list)
        laplace = torch.einsum("tbc,tcd->tbd", laplace, D_tilde)

        return laplace

    def forward(self, input_data, adj_list):

        batch, temporal, cell, feature = input_data.shape

        assert feature == self.input_size
        assert temporal > self.init_length

        output = Variable(
            input_data.data.new(batch, temporal - self.init_length, cell,
                                self.output_size).fill_(0).float())

        laplace_list_forward = self.adj_to_laplace(adj_list)
        laplace_list_backward = self.adj_to_laplace(
            torch.transpose(adj_list, 1, 2))

        hidden = self.init_graph(input_data[:, 0, :, :],
                                 laplace_list_forward[0, :, :])

        for i in range(temporal):

            forward_h = self.forward_gnn(hidden, laplace_list_forward[i, :, :])
            backwad_h = self.backward_gnn(hidden,
                                          laplace_list_backward[i, :, :])

            h_space = torch.cat((forward_h, backwad_h), dim=2)
            h_space = self.sptial_merge(h_space)

            hidden = self.temporal_cell(
                torch.reshape(input_data[:, i, :, :], (batch * cell, feature)),
                torch.reshape(h_space, (batch * cell, self.hidden_size)))
            hidden = hidden.view(batch, cell, self.hidden_size)
            if i >= self.init_length:

                output[:,
                       i - self.init_length, :, :] += self.output_layer(hidden)

        return output

    def set_input_cells(self, input_cells):

        self.input_cells = input_cells

    def infer(self, input_data, adj_list):

        batch, temporal, cell, feature = input_data.shape
        
        predict_cells = [i for i in range(cell) if i not in self.input_cells]

        assert feature == self.input_size
        assert temporal > self.init_length
        assert self.input_cells is not None

        output = Variable(
            input_data.data.new(batch, temporal - self.init_length, cell,
                                self.output_size).fill_(0).float())

        laplace_list_forward = self.adj_to_laplace(adj_list)
        laplace_list_backward = self.adj_to_laplace(
            torch.transpose(adj_list, 1, 2))

        hidden = self.init_graph(input_data[:, 0, :, :],
                                 laplace_list_forward[0, :, :])

        inputs = input_data[:, 0, :, :]

        for i in range(temporal):

            forward_h = self.forward_gnn(hidden, laplace_list_forward[i, :, :])
            backwad_h = self.backward_gnn(hidden,
                                          laplace_list_backward[i, :, :])

            h_space = torch.cat((forward_h, backwad_h), dim=2)
            h_space = self.sptial_merge(h_space)

            hidden = self.temporal_cell(
                torch.reshape(inputs, (batch * cell, feature)),
                torch.reshape(h_space, (batch * cell, self.hidden_size)))
            hidden = hidden.view(batch, cell, self.hidden_size)

            if i >= self.init_length:

                output[:, i - self.init_length, predict_cells, :] += self.output_layer(hidden[:, predict_cells, :])
                output[:, i - self.init_length, self.input_cells, :] += input_data[:, i, self.input_cells, :self.output_size]

                inputs[:, :, :self.output_size] = output[:, i - self.init_length, :, :]
                inputs[:, :, self.output_size:] = input_data[:, 0, :, self.output_size:]

            else:

                inputs += input_data[:, i + 1, :, :]

        return output


if __name__ == "__main__":

    args = {}
    args["input_size"] = 13
    args["output_size"] = 6

    input_data = torch.rand(17, 8, 40, 13)
    adj_list = torch.rand(8, 40, 40)

    model = GCN_GRU(args)
    model.set_input_cells([0, 1, 2, 3])
    print('# generator parameters:',
          sum(param.numel() for param in model.parameters()))

    output = model(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
    output = model.infer(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
