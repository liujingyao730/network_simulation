import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import gcn, gat, dcn
from utils import adj_to_laplace
from coder_model import feature_embedding

class non_dir_model(nn.Module):

    def __init__(self, args):

        super(non_dir_model, self).__init__()

        self.dest_size = args["output_size"]
        self.input_size = args["input_size"] - 3 * self.dest_size
        self.hidden_size = args.get("hidden_size", 64)
        self.init_length = args.get("init_length", 4)
        self.input_cells = None

        self.gnn_type = args.get("gnn", "gcn")
        self.feature_embedding_layer = feature_embedding(
            {"input_size":self.input_size, "output_size":self.dest_size, "struct_size":2*self.dest_size},
            self.gnn_type)
        if self.gnn_type == "gcn":
            self.init_graph = gcn(self.input_size, self.hidden_size)
            self.forward_gnn = gcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = gcn(self.hidden_size, self.hidden_size)
        elif self.gnn_type == "gat":
            self.init_graph = gat(self.input_size, self.hidden_size)
            self.forward_gnn = gat(self.hidden_size, self.hidden_size)
            self.backward_gnn = gat(self.hidden_size, self.hidden_size)
        elif self.gnn_type == "dcn":
            self.init_graph = dcn(self.input_size, self.hidden_size)
            self.forward_gnn = dcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = dcn(self.hidden_size, self.hidden_size)
        else:
            raise NotImplementedError
        self.sptial_merge = nn.Linear(2*self.hidden_size, self.hidden_size)
            
        self.rnn_type = args.get("rnn", "gru")
        if self.rnn_type == "gru":
            self.temporal_cell = nn.GRUCell(2 * self.dest_size, self.hidden_size)
        else:
            raise NotImplementedError
        
        self.output_layer = nn.Linear(self.hidden_size, self.dest_size)
        self.softmax = nn.Softmax(dim=2)

        self.feature_selected = [i for i in range(args["input_size"]) if i < self.dest_size or i >= self.dest_size*4]

    def set_input_cells(self, input_cells):

        self.input_cells = input_cells
    
    def forward(self, inputs_data, adj_list):

        input_data = inputs_data[:, :, :, self.feature_selected]
        batch, temporal, cell, feature = input_data.shape

        assert feature == self.input_size
        assert temporal - 1 > self.init_length
        assert self.input_cells is not None

        predict_cells = [i for i in range(cell) if i not in self.input_cells]

        output = Variable(input_data.data.new(batch, temporal - self.init_length - 1, cell, self.dest_size).fill_(0).float())

        laplace_list_forward = adj_to_laplace(adj_list)
        laplace_list_backward = adj_to_laplace(torch.transpose(adj_list, 1, 2))

        hidden = self.init_graph(input_data[:, 0, :, :], laplace_list_forward[0, :, :])

        for i in range(temporal - 1):

            tmp_input = self.feature_embedding_layer(input_data[:, i, :, :], laplace_list_forward[i, :, :], laplace_list_backward[i, :, :])

            forward_h = self.forward_gnn(hidden, laplace_list_forward[i, :, :])
            backward_h = self.backward_gnn(hidden, laplace_list_backward[i, :, :])

            h_space = self.sptial_merge(torch.cat((forward_h, backward_h), dim=2))

            hidden = self.temporal_cell(
                torch.reshape(tmp_input, (batch * cell, 2 * self.dest_size)),
                torch.reshape(h_space, (batch * cell, self.hidden_size))
            )
            hidden = hidden.view(batch, cell, self.hidden_size)

            if i >= self.init_length:
                output[:, i-self.init_length, predict_cells, :] += self.output_layer(hidden[:, predict_cells, :])
                output[:, i-self.init_length, self.input_cells, :] += input_data[:, i+1, self.input_cells, :self.dest_size]
        
        return output
    
    def infer(self, inputs_data, adj_list):

        input_data = inputs_data[:, :, :, self.feature_selected]
        batch, temporal, cell, feature = input_data.shape

        assert feature == self.input_size
        assert temporal - 1 > self.init_length
        assert self.input_cells is not None

        predict_cells = [i for i in range(cell) if i not in self.input_cells]

        output = Variable(input_data.data.new(batch, temporal - self.init_length - 1, cell, self.dest_size).fill_(0).float())

        laplace_list_forward = adj_to_laplace(adj_list)
        laplace_list_backward = adj_to_laplace(torch.transpose(adj_list, 1, 2))

        hidden = self.init_graph(input_data[:, 0, :, :], laplace_list_forward[0, :, :])

        inputs = input_data[:, 0, :, :]

        for i in range(temporal - 1):

            tmp_input = self.feature_embedding_layer(inputs, laplace_list_forward[i, :, :], laplace_list_backward[i, :, :])

            forward_h = self.forward_gnn(hidden, laplace_list_forward[i, :, :])
            backward_h = self.backward_gnn(hidden, laplace_list_backward[i, :, :])

            h_space = self.sptial_merge(torch.cat((forward_h, backward_h), dim=2))

            hidden = self.temporal_cell(
                torch.reshape(tmp_input, (batch * cell, 2 * self.dest_size)),
                torch.reshape(h_space, (batch * cell, self.hidden_size))
            )
            hidden = hidden.view(batch, cell, self.hidden_size)

            inputs = inputs * 0
            if i >= self.init_length:

                output[:, i - self.init_length, predict_cells, :] += self.output_layer(hidden[:, predict_cells, :])
                output[:, i - self.init_length, self.input_cells, :] += input_data[:, i+1, self.input_cells, :self.dest_size]
                
                if i < temporal - 1:
                    inputs[:, :, :self.dest_size] += output[:, i - self.init_length, :, :]
                    inputs[:, :, self.dest_size:] += input_data[:, i+1, :, self.dest_size:]
            
            else:

                inputs += input_data[:, i+1, :, :]

        return output


if __name__ == "__main__":

    args = {}
    args["input_size"] = 46
    args["output_size"] = 8
    args["gnn"] = "gcn"

    input_data = Variable(torch.rand(17, 8, 40, 46))
    adj_list = Variable(torch.rand(8, 40, 40))

    model = non_dir_model(args)
    model.set_input_cells([0, 1, 2, 3])
    print('# generator parameters:',
          sum(param.numel() for param in model.parameters()))

    output = model(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
    output = model.infer(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
