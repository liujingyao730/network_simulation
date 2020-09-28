import xml.etree.cElementTree as etree
import pandas as pd
import os 
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import sparse
import xml.dom.minidom as dom
import yaml
import pickle
import time

from interval import Interval, IntervalSet
import networkx as nx
import pylab

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
        self.tlc_time = net_information["tlc_time"]
        self.destination = destination
        self.intput_size = len(self.destination)

        self.cell_list = list(set(self.junction_cell.values()))
        i = 0
        self.tlc_index = {self.cell_list[i]:i for i in range(len(self.cell_list))}
        for edge in self.ordinary_cell:
            self.cell_list.extend(self.ordinary_cell[edge]["cell_id"])
        self.cell_num = len(self.cell_list)
        self.cell_index = {self.cell_list[i]:i for i in range(self.cell_num)}
        self.dest_index = {dest:self.cell_list.index(dest) for dest in self.destination}
        self.dest_order = {destiantion[i]:i for i in range(len(destiantion))}

        self.generate_basic_adj()

        self.generate_intervals()

        self.generate_network_feature()

        self.generate_adj(0)

        a = 0

    def generate_basic_adj(self):

        self.basic_adj_row = [self.cell_index[from_cell] for from_cell in self.connection.keys()]
        self.basic_adj_col = [self.cell_index[self.connection[from_cell]] for from_cell in self.connection.keys()]
        self.basic_adj_row.extend(list(range(self.cell_num)))
        self.basic_adj_col.extend(list(range(self.cell_num)))

        self.basic_adj = SparseTensor(
                                row=torch.tensor(self.basic_adj_row),
                                col=torch.tensor(self.basic_adj_col),
                                sparse_sizes=(self.cell_num, self.cell_num)
                            )

        for from_cell in self.signal_connection.keys():
            for to_cell in self.signal_connection[from_cell].keys():
                inter_cell = self.signal_connection[from_cell][to_cell][2]
                self.basic_adj_row.append(self.cell_index[from_cell])
                self.basic_adj_col.append(self.tlc_index[inter_cell])
                self.basic_adj_row.append(self.tlc_index[inter_cell])
                self.basic_adj_col.append(self.cell_index[to_cell])
        
        self.basic_adj_col = torch.tensor(self.basic_adj_col)
        self.basic_adj_row = torch.tensor(self.basic_adj_row)

    def generate_intervals(self):

        self.intervals = [[] for i in range(len(self.tlc_time))]
        self.local_adj = [[] for i in range(len(self.tlc_time))]

        for from_cell in self.signal_connection.keys():
            for to_cell in self.signal_connection[from_cell].keys():
                tlc_index = self.tlc_index[self.signal_connection[from_cell][to_cell][2]]
                start = self.signal_connection[from_cell][to_cell][0]
                end = self.signal_connection[from_cell][to_cell][1]
                self.intervals[tlc_index].append(Interval(start, end))
                self.local_adj[tlc_index].append(
                    SparseTensor(
                        row=torch.tensor([self.cell_index[from_cell], tlc_index]),
                        col=torch.tensor([tlc_index, self.cell_index[to_cell]]),  
                        sparse_sizes=(self.cell_num, self.cell_num)
                    )
                )
    
    def generate_network_feature(self):

        row = self.basic_adj_row.numpy()
        col = self.basic_adj_col.numpy()

        edges = [(row[i], col[i]) for i in range((row.shape[0]))]

        G = nx.DiGraph()
        N = nx.path_graph(self.cell_num)
        G.add_nodes_from(N)
        G.add_edges_from(edges)

        self.network_feature = torch.zeros(self.cell_num, len(self.dest_index)) - 1

        for i in range(self.cell_num):
            for to_cell in self.dest_index.keys():
                j = self.dest_order[to_cell]
                target = self.dest_index[to_cell]
                try:
                    self.network_feature[i][j] += nx.dijkstra_path_length (G, source=i, target=target)
                except nx.exception.NetworkXNoPath:
                    pass
        '''
        pos=nx.spring_layout(G)
        nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5 )
        pylab.title('topology',fontsize=15)
        pylab.show()
        '''     

    def generate_adj(self, time):

        time_list = [(time + signal[1]) % signal[0] for signal in self.tlc_time]

        self.adj = self.basic_adj

        for i in range(len(time_list)):
            t = time_list[i]
            for j in range(len(self.intervals[i])):
                if t in self.intervals[i][j]:
                    self.adj += self.local_adj[i][j]


    

with open("test_net.pkl", 'rb') as f:
    net_information = pickle.load(f)
destiantion = [end+'-2' for end in utils.end_edge.keys()]
a = network(net_information, None, destiantion)
