import os
import torch
import torch.nn as nn
from torch.autograd import Variable

from Gman import GMAN
import dir_manage as d


class Gman_sim(nn.Module):

    def __init__(self, SE_file, args, bn_decay):
        super().__init__()

        with open(SE_file, 'r') as f:
            lines = f.readlines()
            temp = lines[0].split(' ')
            num_vertex, dims = int(temp[0]), int(temp[1])
            SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
            for line in lines[1:]:
                temp = line.split(' ')
                index = int(temp[0])
                SE[index] = torch.tensor([float(ch) for ch in temp[1:]])

        self.gman = GMAN(SE, args, bn_decay)
        self.history_length = args['num_his']
        self.init_length = args["init_length"]
        self.output_size = args["output_size"]
        self.input_size = args["input_size"]
        self.input_cells = None
        self.cycle_time = args.get("cycle", 90)
        
        self.base_TE = torch.cat(
            (
                torch.zeros(self.history_length+1, 1),
                torch.arange(self.history_length+1).unsqueeze(1)
            ), dim=1
        )
        self.base_TE.unsqueeze_(0)
    
    def set_input_cells(self, input_cells):

        self.input_cells = input_cells

    def infer(self, X, time, mod="infer"):

        assert mod == "train" or mod == "infer"
        assert self.input_cells

        batch, temporal, cell, feature = X.shape

        assert feature == self.input_size
        assert temporal - 1 > self.init_length

        predict_cells = [i for i in range(cell) if i not in self.input_cells]

        output = Variable(X.data.new(batch, temporal-self.init_length-1, cell, self.output_size).fill_(0).float())

        if self.history_length <= self.init_length:
            input = X[:, self.init_length-self.history_length:self.init_length, :, :]
        else:
            input = torch.cat(
                (
                    Variable(X.data.new(batch, self.history_length-self.init_length, cell, feature).fill_(0).float()),
                    X[:, :self.init_length, :, :]
                ), dim=1
            )

        for i in range(temporal-self.init_length-1):
            
            TE = self.base_TE
            TE[:, :, 1] += (time % self.cycle_time)
            TE = TE.expand(batch, -1, -1)

            tmp_output = self.gman(input, TE)

            output[:, i, :, :] += tmp_output.squeeze(1)

            input[:, :-1, :, :] = input[:, 1:, :, :]
            input[:, -1, :, self.output_size:] = X[:, i, :, self.output_size:]

            if mod == "train":
                input[:, -1, :, :self.output_size] = X[:, i, :, :self.output_size]
            else:
                input[:, -1, :, :self.output_size] = output[:, i, :, :]
        
        return output

    def forward(self, X, time):
        
        return self.infer(X, time, "train")


if __name__ == "__main__":

    SE_file = os.path.join(d.cell_data_path, 'four_large_SE.txt')
    bn_decay = 0.1
    args = {}
    args['num_his'] = 4
    args['init_length'] = 4
    args['output_size'] = 8
    args['input_size'] = 46
    args['L'] = 1
    args['K'] = 8
    args['d'] = 8
    
    model = Gman_sim(SE_file, args, bn_decay)
    model.set_input_cells([1, 2, 3, 4, 5, 6, 7, 8])
    
    input_data = torch.rand((5, 24, 110, 46))
    time = 100
    
    loss = torch.sum(model(input_data, time))
    loss.backward()
    loss = torch.sum(model.infer(input_data, time))
    loss.backward()
