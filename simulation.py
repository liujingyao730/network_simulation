import xml.etree.cElementTree as etree
import pandas as pd
import os 
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import xml.dom.minidom as dom
import yaml
import pickle

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul

import utils


class network(object):

    def __init__(self, net_information, simulator, destination):

        self.simulator = simulator
        self.ordinary_cell = net_information["ordinary_cell"]
        self.junction_cell = net_information["junction_cell"]
        self.connection = net_information["connection"]
        self.signal_connection = net_information["signal_connection"]
        self.cycle = net_information["cycle"]
        self.destination = destination
        self.intput_size = len(self.destination)

        self.cell_list = list(set(self.junction_cell.values()))
        for edge in self.ordinary_cell:
            self.cell_list.extend(self.ordinary_cell[edge]["cell_id"])
        self.cell_num = len(self.cell_list)
        self.cell_index = {self.cell_list[i]:i for i in range(self.cell_num)}
        self.dest_index = {dest:self.cell_list.index(dest) for dest in self.destination}

        self.basic_adj_row = [self.cell_index[from_cell] for from_cell in self.connection.keys()]
        self.basic_adj_col = [self.cell_index[self.connection[from_cell]] for from_cell in self.connection.keys()]

        for from_cell in self.signal_connection.keys():
            for to_cell in self.signal_connection[from_cell].keys():
                self.basic_adj_row.append(self.cell_index[from_cell])
                self.basic_adj_col.append(self.cell_index[to_cell])
        self.basic_adj_row.extend(list(range(self.cell_num)))
        self.basic_adj_col.extend(list(range(self.cell_num)))
        self.basic_adj_col = torch.tensor(self.basic_adj_col)
        self.basic_adj_row = torch.tensor(self.basic_adj_row)

        self.basic_adj = SparseTensor(row=self.basic_adj_row, col=self.basic_adj_col, sparse_sizes=(self.cell_num, self.cell_num))
        
        a = 1



with open("test_net.pkl", 'rb') as f:
    net_information = pickle.load(f)
destiantion = [end+'-2' for end in utils.end_edge.keys()]
a = network(net_information, None, destiantion)
