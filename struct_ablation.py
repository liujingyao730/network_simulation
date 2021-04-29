import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import gcn, gat, dcn
from utils import adj_to_laplace
from coder_model import attention_on_node

class single_attention(nn.Module):

    def __init__(self, args):

        super(single_attention, self).__init__()

        self.dest_size = args["output_size"]
        self.input_size = args["input_size"]
        self.hidden_size = args.get("hidden_size", 64)
        self.init_length = args.get("init_length", 4)
        self.input_cells = None
        self.gnn_type = args.get("gnn_type", "gcn")

        self.node_embedding_layer = attention_on_node({
            "output_size": self.dest_size,
            "encoding_size": self.dest_size*4,
            "gnn": self.gnn_type
        })

        self.gnn_type = args.get("gnn", "gcn")
        if self.gnn_type == "gcn":
            self.init_graph = gcn(self.input_size, self.hidden_size)
            self.forward_gnn = gcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = gcn(self.hidden_size, self.hidden_size)
        elif self.gnn_type == "gat":
            
            raise NotImplementedError

        elif self.gnn_type == "dcn":
            self.init_graph = dcn(self.input_size, self.hidden_size)
            self.forward_gnn = dcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = dcn(self.hidden_size, self.hidden_size)
        else:
            raise NotImplementedError
        self.sptial_merge = nn.Linear(2*self.hidden_size, self.hidden_size)
        
        self.state_gate = nn.Linear(self.input_size-5*self.dest_size, 2*self.dest_size)
        self.sigmoid = nn.Sigmoid()

        self.rnn_type = args.get("rnn", "gru")
        if self.rnn_type == "gru":
            self.temporal_cell = nn.GRUCell(2*self.dest_size, self.hidden_size)
        elif self.rnn_type == "lstm":
            self.temporal_cell = nn.LSTMCell(2*self.dest_size, self.hidden_size)
        elif self.rnn_type == "conv":
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.output_layer = nn.Linear(self.hidden_size, self.dest_size)

    def set_input_cells(self, input_cells):

        self.input_cells = input_cells
    
    def infer(self, input_data, adj_list, mod="infer"):

        assert mod == "train" or mod == "infer"
        assert self.input_cells is not None

        batch, temporal, cell, feature = input_data.shape

        assert feature == self.input_size
        assert temporal - 1 > self.init_length

        predict_cells = [i for i in range(cell) if i not in self.input_cells]

        output = Variable(input_data.data.new(batch, temporal-self.init_length-1, cell, self.dest_size).fill_(0).float())

        laplace_list_forward = adj_to_laplace(adj_list)
        laplace_list_backward = adj_to_laplace(torch.transpose(adj_list, 1, 2))

        hidden = self.init_graph(input_data[:, 0, :, :], laplace_list_forward[0, :, :])
        if self.rnn_type == "lstm":
            c = Variable(input_data.data.new(batch * cell, self.hidden_size).fill_(0).float())

        inputs = input_data[:, 0, :, :]

        for i in range(temporal-1):

            dyn_feature = inputs[:, :, :self.dest_size]
            embedding_feature = inputs[:, :, self.dest_size:5*self.dest_size]
            state_feature = inputs[:, :, 5*self.dest_size:]

            tmp_input = self.node_embedding_layer(
                embedding_feature, dyn_feature,
                laplace_list_forward[i, :, :], laplace_list_backward[i, :, :]
            )

            state_info = self.state_gate(state_feature)
            state_info = self.sigmoid(state_info)
            tmp_input = tmp_input.mul(state_info)

            forward_h = self.forward_gnn(hidden, laplace_list_forward[i, :, :])
            backward_h = self.backward_gnn(hidden, laplace_list_backward[i, :, :])

            h_space = self.sptial_merge(torch.cat((forward_h, backward_h), dim=2))

            if self.rnn_type == "gru":
                hidden = self.temporal_cell(
                    torch.reshape(tmp_input, (batch * cell, 2 * self.dest_size)),
                    torch.reshape(h_space, (batch * cell, self.hidden_size))
                )
            elif self.rnn_type == "lstm":
                hidden, c = self.temporal_cell(
                    torch.reshape(tmp_input, (batch * cell, 2 * self.dest_size)),
                    (
                        torch.reshape(h_space, (batch * cell, self.hidden_size)),
                        c
                    )
                )
                
            hidden = hidden.view(batch, cell, self.hidden_size)

            inputs = inputs * 0

            if i >= self.init_length:
                
                output[:, i - self.init_length, predict_cells, :] += self.output_layer(hidden[:, predict_cells, :])
                output[:, i - self.init_length, self.input_cells, :] += input_data[:, i+1, self.input_cells, :self.dest_size]

                if mod == "infer" and i < temporal - 1:
                    inputs[:, :, :self.dest_size] += output[:, i - self.init_length, :, :]
                    inputs[:, :, self.dest_size:] += input_data[:, i+1, :, self.dest_size:]
                else:
                    inputs += input_data[:, i+1, :, :]
                
            else:
                inputs += input_data[:, i+1, :, :]

        return output
    
    def forward(self, input_data, adj_list):

        return self.infer(input_data, adj_list, mod="train")


class single_attention_non_gate(nn.Module):

    def __init__(self, args):

        super(single_attention_non_gate, self).__init__()

        self.dest_size = args["output_size"]
        self.input_size = args["input_size"]
        self.hidden_size = args.get("hidden_size", 64)
        self.init_length = args.get("init_length", 4)
        self.input_cells = None
        self.gnn_type = args.get("gnn_type", "gcn")

        self.node_embedding_layer = attention_on_node({
            "output_size": self.dest_size,
            "encoding_size": self.input_size - self.dest_size,
            "gnn": self.gnn_type
        })

        self.gnn_type = args.get("gnn", "gcn")
        if self.gnn_type == "gcn":
            self.init_graph = gcn(self.input_size, self.hidden_size)
            self.forward_gnn = gcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = gcn(self.hidden_size, self.hidden_size)
        elif self.gnn_type == "gat":
            raise NotImplementedError
        elif self.gnn_type == "dcn":
            self.init_graph = dcn(self.input_size, self.hidden_size)
            self.forward_gnn = dcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = dcn(self.hidden_size, self.hidden_size)
        else:
            raise NotImplementedError
        self.sptial_merge = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.rnn_type = args.get("rnn", "gru")
        if self.rnn_type == "gru":
            self.temporal_cell = nn.GRUCell(2*self.dest_size, self.hidden_size)
        elif self.rnn_type == "lstm":
            self.temporal_cell = nn.LSTMCell(2*self.dest_size, self.hidden_size)
        elif self.rnn_type == "conv":
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.output_layer = nn.Linear(self.hidden_size, self.dest_size)

    def set_input_cells(self, input_cells):

        self.input_cells = input_cells
    
    def infer(self, input_data, adj_list, mod="infer"):

        assert mod == "train" or mod == "infer"
        assert self.input_cells is not None

        batch, temporal, cell, feature = input_data.shape

        assert feature == self.input_size
        assert temporal - 1 > self.init_length

        predict_cells = [i for i in range(cell) if i not in self.input_cells]

        output = Variable(input_data.data.new(batch, temporal-self.init_length-1, cell, self.dest_size).fill_(0).float())

        laplace_list_forward = adj_to_laplace(adj_list)
        laplace_list_backward = adj_to_laplace(torch.transpose(adj_list, 1, 2))

        hidden = self.init_graph(input_data[:, 0, :, :], laplace_list_forward[0, :, :])
        if self.rnn_type == "lstm":
            c = Variable(input_data.data.new(batch * cell, self.hidden_size).fill_(0).float())

        inputs = input_data[:, 0, :, :]

        for i in range(temporal-1):

            dyn_feature = inputs[:, :, :self.dest_size]
            embedding_feature = inputs[:, :, self.dest_size:]

            tmp_input = self.node_embedding_layer(
                embedding_feature, dyn_feature,
                laplace_list_forward[i, :, :], laplace_list_backward[i, :, :]
            )

            forward_h = self.forward_gnn(hidden, laplace_list_forward[i, :, :])
            backward_h = self.backward_gnn(hidden, laplace_list_backward[i, :, :])

            h_space = self.sptial_merge(torch.cat((forward_h, backward_h), dim=2))

            if self.rnn_type == "gru":
                hidden = self.temporal_cell(
                    torch.reshape(tmp_input, (batch * cell, 2 * self.dest_size)),
                    torch.reshape(h_space, (batch * cell, self.hidden_size))
                )
            elif self.rnn_type == "lstm":
                hidden, c = self.temporal_cell(
                    torch.reshape(tmp_input, (batch * cell, 2 * self.dest_size)),
                    (
                        torch.reshape(h_space, (batch * cell, self.hidden_size)),
                        c
                    )
                )
                
            hidden = hidden.view(batch, cell, self.hidden_size)

            inputs = inputs * 0

            if i >= self.init_length:
                
                output[:, i - self.init_length, predict_cells, :] += self.output_layer(hidden[:, predict_cells, :])
                output[:, i - self.init_length, self.input_cells, :] += input_data[:, i+1, self.input_cells, :self.dest_size]

                if mod == "infer" and i < temporal - 1:
                    inputs[:, :, :self.dest_size] += output[:, i - self.init_length, :, :]
                    inputs[:, :, self.dest_size:] += input_data[:, i+1, :, self.dest_size:]
                else:
                    inputs += input_data[:, i+1, :, :]
                
            else:
                inputs += input_data[:, i+1, :, :]

        return output
    
    def forward(self, input_data, adj_list):

        return self.infer(input_data, adj_list, mod="train")


class baseline(nn.Module):

    def __init__(self, args):

        super(baseline, self).__init__()

        self.dest_size = args["output_size"]
        self.input_size = args["input_size"]
        self.hidden_size = args.get("hidden_size", 64)
        self.init_length = args.get("init_length", 4)
        self.input_cells = None
        self.gnn_type = args.get("gnn_type", "gcn")

        self.gnn_type = args.get("gnn", "gcn")
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
            self.temporal_cell = nn.GRUCell(self.input_size, self.hidden_size)
        elif self.rnn_type == "lstm":
            self.temporal_cell = nn.LSTMCell(self.input_size, self.hidden_size)
        elif self.rnn_type == "conv":
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.output_layer = nn.Linear(self.hidden_size, self.dest_size)

    def set_input_cells(self, input_cells):

        self.input_cells = input_cells
    
    def infer(self, input_data, adj_list, index=None, weight=None, mod="infer"):

        assert mod == "train" or mod == "infer"
        assert self.input_cells is not None

        batch, temporal, cell, feature = input_data.shape

        assert feature == self.input_size
        assert temporal - 1 > self.init_length
        
        if index is not None:
            index_list, reverse_index_list = index
            weight_list, reverse_weight_list = weight

        predict_cells = [i for i in range(cell) if i not in self.input_cells]

        output = Variable(input_data.data.new(batch, temporal-self.init_length-1, cell, self.dest_size).fill_(0).float())

        laplace_list_forward = adj_to_laplace(adj_list)
        laplace_list_backward = adj_to_laplace(torch.transpose(adj_list, 1, 2))

        if self.gnn_type == "gat":
            hidden = self.init_graph(input_data[:, 0, :, :], index_list[0, :, :], weight_list[0, :, :])
        else:
            hidden = self.init_graph(input_data[:, 0, :, :], laplace_list_forward[0, :, :])
        if self.rnn_type == "lstm":
            c = Variable(input_data.data.new(batch * cell, self.hidden_size).fill_(0).float())

        inputs = input_data[:, 0, :, :]

        for i in range(temporal-1):

            if self.gnn_type == "gat":
                forward_h = self.forward_gnn(hidden, index_list[i, :, :], weight_list[i, :, :])
                backward_h = self.backward_gnn(hidden, reverse_index_list[i, :, :], reverse_weight_list[i, :, :])
            else:
                forward_h = self.forward_gnn(hidden, laplace_list_forward[i, :, :])
                backward_h = self.backward_gnn(hidden, laplace_list_backward[i, :, :])

            h_space = self.sptial_merge(torch.cat((forward_h, backward_h), dim=2))

            if self.rnn_type == "gru":
                hidden = self.temporal_cell(
                    torch.reshape(inputs, (batch * cell, self.input_size)),
                    torch.reshape(h_space, (batch * cell, self.hidden_size))
                )
            elif self.rnn_type == "lstm":
                hidden, c = self.temporal_cell(
                    torch.reshape(inputs, (batch * cell, self.input_size)),
                    (
                        torch.reshape(h_space, (batch * cell, self.hidden_size)),
                        c
                    )
                )
                
            hidden = hidden.view(batch, cell, self.hidden_size)

            inputs = inputs * 0

            if i >= self.init_length:
                
                output[:, i - self.init_length, predict_cells, :] += self.output_layer(hidden[:, predict_cells, :])
                output[:, i - self.init_length, self.input_cells, :] += input_data[:, i+1, self.input_cells, :self.dest_size]

                if mod == "infer" and i < temporal - 1:
                    inputs[:, :, :self.dest_size] += output[:, i - self.init_length, :, :]
                    inputs[:, :, self.dest_size:] += input_data[:, i+1, :, self.dest_size:]
                else:
                    inputs += input_data[:, i+1, :, :]
                
            else:
                inputs += input_data[:, i+1, :, :]

        return output
    
    def forward(self, input_data, adj_list, index=None, weight=None):

        return self.infer(input_data, adj_list, index, weight, mod="train")


if __name__ == "__main__":

    args = {}
    args["input_size"] = 46
    args["output_size"] = 8
    args["hidden_size"] = 64
    args["gnn"] = "gat"
    args["rnn"] = "gru"

    input_data = Variable(torch.rand(17, 8, 40, 46))
    adj_list = Variable(torch.rand(8, 40, 40))

    model = baseline(args)
    model.set_input_cells([0, 1, 2, 3])
    print('# generator parameters:',
          sum(param.numel() for param in model.parameters()))

    output = model(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
    output = model.infer(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
