import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import gcn, gat, dcn
from utils import adj_to_laplace

class feature_embedding(nn.Module):

    def __init__(self, args, gnn="gcn"):

        super(feature_embedding, self).__init__()

        self.dest_size = args["output_size"]
        self.input_size = args["input_size"]
        self.struct_size = args["struct_size"]

        if gnn == "gcn":
            self.dest_encoder_forward = gcn(self.struct_size-self.dest_size, self.dest_size)
            self.dest_encoder_backward = gcn(self.struct_size-self.dest_size, self.dest_size)
        else:
            raise NotImplementedError

        self.feature_embedding_layer = torch.nn.Linear(self.input_size-self.struct_size, 2 * self.dest_size)
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, node_feautre, adj_foward, adj_backward):

        batch_size, N, feature = node_feautre.shape

        assert feature == self.input_size

        dyn_feature = node_feautre[:, :, :self.dest_size]
        struct_feature = node_feautre[:, :, self.dest_size:self.struct_size]
        stat_feature = node_feautre[:, :, self.struct_size:]

        dist_code_forward = self.dest_encoder_forward(struct_feature, adj_foward)
        dist_code_backward = self.dest_encoder_backward(struct_feature, adj_backward)
        forward_code = torch.mul(dyn_feature, dist_code_forward)
        backward_code = torch.mul(dyn_feature, dist_code_backward)
        input_embedding = torch.cat((forward_code, backward_code), dim=2)

        stat_code = self.feature_embedding_layer(stat_feature)
        stat_code = self.sigmoid(stat_code)
        input_embedding = torch.mul(input_embedding, stat_code)

        return input_embedding

class st_node_encoder(nn.Module):

    def __init__(self, args):

        super(st_node_encoder, self).__init__()

        self.dest_size = args["output_size"]
        self.input_size = args["input_size"]
        self.hidden_size = args.get("hidden_size", 64)
        self.init_length = args.get("init_length", 4)
        self.input_cells = None
        args["struct_size"] = 5 * self.dest_size

        self.gnn_type = args.get("gnn", "gcn")
        self.feature_embedding_layer = feature_embedding(
            args, self.gnn_type
        )
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
    
    def set_input_cells(self, input_cells):

        self.input_cells = input_cells
    
    def forward(self, input_data, adj_list):

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
    
    def infer(self, input_data, adj_list):

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

class attention_on_node(nn.Module):

    def __init__(self, args):

        super(attention_on_node, self).__init__()

        self.dest_size = args["output_size"]
        self.encoding_size = args["encoding_size"]
        self.gnn_type = args.get("gnn", "gcn")
        
        if self.gnn_type == "gcn":
            self.encoder_forward = gcn(self.encoding_size, self.dest_size)
            self.encoder_backward = gcn(self.encoding_size, self.dest_size)
        elif self.gnn_type == "gat":
            self.encoder_forward = gat(self.encoding_size, self.dest_size)
            self.encoder_backward = gat(self.encoding_size, self.dest_size)
        elif self.gnn_type == "dcn":
            self.encoder_backward = dcn(self.encoding_size, self.dest_size)
            self.encoder_forward = dcn(self.encoding_size, self.dest_size)
        else:
            raise NotImplementedError

        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, inputs, dyn_feature, adj_foward, adj_backward):

        assert inputs.shape[2] == self.encoding_size
        assert dyn_feature.shape[2] == self.dest_size

        alpha_forward = self.encoder_forward(inputs, adj_foward)
        alpha_backward = self.encoder_backward(inputs, adj_backward)
        att_forward = self.softmax(alpha_forward)
        att_backward = self.softmax(alpha_backward)
        
        forward_code = torch.mul(dyn_feature, att_forward)
        backward_code = torch.mul(dyn_feature, att_backward)

        return torch.cat((forward_code, backward_code), dim=2)


class coder_on_dir(nn.Module):

    def __init__(self, args):

        super(coder_on_dir, self).__init__()

        self.dest_size = args["output_size"]
        self.input_size = args["input_size"]
        self.hidden_size = args.get("hidden_size", 64)
        self.init_length = args.get("init_length", 4)
        self.input_cells = None

        self.lane_number_embedding_layer = attention_on_node({
            "output_size": self.dest_size,
            "encoding_size": self.dest_size
        })
        self.dir_embedding_layer = attention_on_node({
            "output_size": self.dest_size,
            "encoding_size": self.dest_size*3
        })

        self.gnn_type = args.get("gnn", "gcn")
        if self.gnn_type == "gcn":
            self.init_graph = gcn(self.input_size, self.hidden_size)
            self.forward_gnn = gcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = gcn(self.hidden_size, self.hidden_size)
        else:
            raise NotImplementedError
        self.sptial_merge = nn.Linear(2*self.hidden_size, self.hidden_size)
        
        self.state_gate = nn.Linear(self.input_size-5*self.dest_size, 4*self.dest_size)
        self.sigmoid = nn.Sigmoid()

        self.rnn_type = args.get("rnn", "gru")
        if self.rnn_type == "gru":
            self.temporal_cell = nn.GRUCell(4*self.dest_size, self.hidden_size)
        else:
            raise NotImplementedError

        self.output_layer = nn.Linear(self.hidden_size, self.dest_size)

    def set_input_cells(self, input_cells):

        self.input_cells = input_cells

    def forward(self, input_data, adj_list):

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

            dyn_feature = input_data[:, i, :, :self.dest_size]
            lane_number = input_data[:, i, :, self.dest_size:2*self.dest_size]
            dir_feature = input_data[:, i, :, 2*self.dest_size:5*self.dest_size]
            state_feature = input_data[:, i, :, 5*self.dest_size:]

            lane_number_embedding = self.lane_number_embedding_layer(
                lane_number, dyn_feature,
                laplace_list_forward[i, :, :], laplace_list_backward[i, :, :]
            )
            dir_embedding = self.dir_embedding_layer(
                dir_feature, dyn_feature,
                laplace_list_forward[i, :, :], laplace_list_backward[i, :, :]
            )
            tmp_input = torch.cat((lane_number_embedding, dir_embedding), dim=2)

            state_info = self.state_gate(state_feature)
            state_info = self.sigmoid(state_info)
            tmp_input = torch.mul(tmp_input, state_info)

            forward_h = self.forward_gnn(hidden, laplace_list_forward[i, :, :])
            backward_h = self.backward_gnn(hidden, laplace_list_backward[i, :, :])

            h_space = self.sptial_merge(torch.cat((forward_h, backward_h), dim=2))

            hidden = self.temporal_cell(
                torch.reshape(tmp_input, (batch * cell, 4 * self.dest_size)),
                torch.reshape(h_space, (batch * cell, self.hidden_size))
            )
            hidden = hidden.view(batch, cell, self.hidden_size)

            if i >= self.init_length:
                output[:, i-self.init_length, predict_cells, :] += self.output_layer(hidden[:, predict_cells, :])
                output[:, i-self.init_length, self.input_cells, :] += input_data[:, i+1, self.input_cells, :self.dest_size]

        return output
    
    def infer(self, input_data, adj_list):

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

            dyn_feature = inputs[:, :, :self.dest_size]
            lane_number = inputs[:, :, self.dest_size:2*self.dest_size]
            dir_feature = inputs[:, :, 2*self.dest_size:5*self.dest_size]
            state_feature = inputs[:, :, 5*self.dest_size:]

            lane_number_embedding = self.lane_number_embedding_layer(
                lane_number, dyn_feature,
                laplace_list_forward[i, :, :], laplace_list_backward[i, :, :]
            )
            dir_embedding = self.dir_embedding_layer(
                dir_feature, dyn_feature,
                laplace_list_forward[i, :, :], laplace_list_backward[i, :, :]
            )
            tmp_input = torch.cat((lane_number_embedding, dir_embedding), dim=2)

            state_info = self.state_gate(state_feature)
            state_info = self.sigmoid(state_info)
            tmp_input = torch.mul(tmp_input, state_info)

            forward_h = self.forward_gnn(hidden, laplace_list_forward[i, :, :])
            backward_h = self.backward_gnn(hidden, laplace_list_backward[i, :, :])

            h_space = self.sptial_merge(torch.cat((forward_h, backward_h), dim=2))

            hidden = self.temporal_cell(
                torch.reshape(tmp_input, (batch * cell, 4 * self.dest_size)),
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

    model = coder_on_dir(args)
    model.set_input_cells([0, 1, 2, 3])
    print('# generator parameters:',
          sum(param.numel() for param in model.parameters()))

    output = model(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
    output = model.infer(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
