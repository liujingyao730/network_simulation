import torch
import torch.nn as nn
from torch.autograd import Variable

from layers import gcn, gat, dcn
from utils import adj_to_laplace

class st_block(nn.Module):

    def __init__(self, args):

        super(st_block, self).__init__()

        self.input_size = args["input_size"]
        self.output_size = args.get("output_size", 64)
        self.window = args.get("window", 3)

        padding_size = int(self.window / 2)
        self.conv_1 = nn.Conv1d(self.input_size, self.output_size, kernel_size=self.window, padding=padding_size)
        self.conv_2 = nn.Conv1d(self.input_size, self.output_size, kernel_size=self.window, padding=padding_size)
        self.sigmiod = nn.Sigmoid()

    def forward(self, input_data):
        
        batch_size, temporal, cell, feature = input_data.shape

        assert feature == self.input_size

        inputs = input_data.permute(0, 2, 3, 1).contiguous()
        inputs = inputs.view(batch_size*cell, self.input_size, temporal)

        P = self.conv_1(inputs)
        Q = self.conv_2(inputs)
        output = P * self.sigmiod(Q)

        output = output.view(batch_size, cell, self.output_size, -1)
        output = output.permute(0, 3, 1, 2).contiguous()

        return output


class gnn_conv(nn.Module):

    def __init__(self, args):

        super(gnn_conv, self).__init__()

        self.dest_size = args["output_size"]
        self.input_size = args["input_size"]
        self.hidden_size = args.get("hidden_size", 64)
        self.init_length = args.get("init_length", 4)
        self.window = args.get("window", 3)
        self.input_cells = None

        assert self.window <= self.init_length

        self.gnn_type = args.get("gnn_type", "gcn")
        if self.gnn_type == "gcn":
            self.forward_gnn = gcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = gcn(self.hidden_size, self.hidden_size)
        elif self.gnn_type == "gat":
            self.forward_gnn = gat(self.hidden_size, self.hidden_size)
            self.backward_gnn = gat(self.hidden_size, self.hidden_size)
        elif self.gnn_type == "dcn":
            self.forward_gnn = dcn(self.hidden_size, self.hidden_size)
            self.backward_gnn = dcn(self.hidden_size, self.hidden_size)
        else:
            raise NotImplementedError
        self.sptial_merge = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.conv_1 = st_block(
            {
                "input_size": self.input_size,
                "output_size": self.hidden_size
            }
        )
        self.conv_2 = st_block(
            {
                "input_size": self.hidden_size,
                "output_size": self.hidden_size
            }
        )

        self.output_layer = nn.Linear(self.hidden_size, self.dest_size)
        self.relu = nn.ReLU()
    
    def set_input_cells(self, input_cells):

        self.input_cells = input_cells
    
    def infer(self, input_data, adj_list, mod="infer"):

        assert mod == "infer" or mod == "train"
        assert self.input_cells is not None

        batch, temporal, cell, feature = input_data.shape

        assert feature == self.input_size
        assert temporal - 1 > self.init_length

        predict_cells = [i for i in range(cell) if i not in self.input_cells]

        outputs = Variable(input_data.data.new(batch, temporal-self.init_length-1, cell, self.dest_size).fill_(0).float())

        laplace_list_forward = adj_to_laplace(adj_list)
        laplace_list_backward = adj_to_laplace(torch.transpose(adj_list, 1, 2))

        h_2 = Variable(input_data.data.new(batch, temporal, cell, self.hidden_size).fill_(0).float())

        if mod == "train":

            h_1 = self.conv_1(input_data)

            for i in range(temporal-1):

                forward_h = self.forward_gnn(h_1[:, i, :, :], laplace_list_forward[i, :, :])
                backward_h = self.backward_gnn(h_1[:, i, :, :], laplace_list_backward[i, :, :])
                h_2[:, i, :, :] += self.sptial_merge(torch.cat((forward_h, backward_h), dim=2))
            
            h_2 = self.relu(h_2)
            h_3 = self.conv_2(h_2)

            output = self.output_layer(h_3)
            outputs += output[:, self.init_length+1:, :, :]
        
        else:

            inputs = input_data[:, self.init_length-self.window:self.init_length, :, :]
            h_2 = Variable(input_data.data.new(batch, self.window, cell, self.hidden_size).fill_(0).float())
            pos = int(self.window / 2) + 1

            for i in range(self.init_length - self.window, temporal-1-self.window):

                h_1 = self.conv_1(inputs)
                h_2 = h_2 * 0
                
                for j in range(self.window):
                    
                    forward_h = self.forward_gnn(h_1[:, j, :, :], laplace_list_forward[j, :, :])
                    backward_h = self.backward_gnn(h_1[:, j, :, :], laplace_list_backward[j, :, :])
                    h_2[:, j, :, :] += self.sptial_merge(torch.cat((forward_h, backward_h), dim=2))
                
                h_2 = self.relu(h_2)
                h_3 = self.conv_2(h_2)

                outputs[:, i - self.init_length + self.window, predict_cells, :] += self.output_layer(h_3[:, pos, predict_cells, :])
                outputs[:, i - self.init_length + self.window, self.input_cells, :] += input_data[:, i+1+self.window, self.input_cells, :self.dest_size]

                inputs[:, :2, :, :] = inputs[:, 1:, :, :]
                inputs[:, 2, :, :self.dest_size] = outputs[:, i - self.init_length + self.window, :, :]
                inputs[:, 2, :, self.dest_size:] = input_data[:, i + self.window + 1, :, self.dest_size:]
        
        return outputs

    def forward(self, input_data, adj_list):

        return self.infer(input_data, adj_list, mod="train")


if __name__ == "__main__":

    args = {}
    args["input_size"] = 46
    args["output_size"] = 8
    args["hidden_size"] = 64
    input_data = Variable(torch.rand(30, 28, 40, 46))
    adj_list = Variable(torch.rand(28, 40, 40))
    model = gnn_conv(args)
    model.set_input_cells([0, 1, 2, 3])
    print('# generator parameters:',
          sum(param.numel() for param in model.parameters()))

    output = model(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
    output = model.infer(input_data, adj_list)
    fake_loss = torch.sum(output)
    fake_loss.backward()
