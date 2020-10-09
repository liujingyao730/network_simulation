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
import tqdm
from interval import Interval, IntervalSet
import networkx as nx
import pylab

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from torchnet import meter

import utils
from network_data import network_dataset
from model import test_model

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
        self.dest_num = len(self.destination)
        self.cell_index = {self.cell_list[i]:i for i in range(self.cell_num)}
        self.dest_index = {dest:self.cell_list.index(dest) for dest in self.destination}
        self.dest_order = {destiantion[i]:i for i in range(len(destiantion))}

        self.generate_basic_adj()

        self.generate_intervals()

        self.generate_network_feature()

        self.generate_adj(0)

    def generate_basic_adj(self):

        self.rows = [self.cell_index[cell] for cell in self.connection.keys()]
        self.cols = [self.cell_index[self.connection[cell]] for cell in self.connection.keys()]
        self.vals = [1 for cell in self.connection.keys()]
        for i in range(self.cell_num):
            self.rows.append(i)
            self.cols.append(i)
            self.vals.append(1)

        self.basic_adj = sparse.csc_matrix((self.vals, (self.rows, self.cols)), shape=(self.cell_num, self.cell_num))

        for from_cell in self.signal_connection.keys():
            for to_cell in self.signal_connection[from_cell].keys():
                inter_cell = self.signal_connection[from_cell][to_cell][2]
                from_id = self.cell_index[from_cell]
                to_id = self.cell_index[to_cell]
                inter_id = self.cell_index[inter_cell]
                self.rows.extend([from_id, inter_id])
                self.cols.extend([inter_id, to_id])
                self.vals.extend([1, 1])
        
        self.all_adj = sparse.csc_matrix((self.vals, (self.rows, self.cols)), shape=(self.cell_num, self.cell_num))

    def generate_intervals(self):

        self.intervals = [[] for i in range(len(self.tlc_time))]
        self.local_adj = [[] for i in range(len(self.tlc_time))]

        for from_cell in self.signal_connection.keys():
            for to_cell in self.signal_connection[from_cell].keys():
                inter_id = self.tlc_index[self.signal_connection[from_cell][to_cell][2]]
                start_id = self.cell_index[from_cell]
                to_id = self.cell_index[to_cell]
                start = self.signal_connection[from_cell][to_cell][0]
                end = self.signal_connection[from_cell][to_cell][1]
                self.intervals[inter_id].append([start, end, start_id, inter_id])
                self.local_adj[inter_id].append(
                    sparse.csc_matrix(([1, 1], ([start_id, inter_id], [inter_id, to_id])), shape=(self.cell_num, self.cell_num))
                )
    
    def generate_network_feature(self):

        row = self.rows
        col = self.cols

        edges = [(row[i], col[i]) for i in range(len(row))]

        G = nx.DiGraph()
        N = nx.path_graph(self.cell_num)
        G.add_nodes_from(N)
        G.add_edges_from(edges)

        self.network_feature = torch.zeros(self.cell_num, len(self.dest_index)+1) - 1

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
                if self.intervals[i][j][0] <= t <= self.intervals[i][j][1]:
                    self.adj += self.local_adj[i][j]
                    self.network_feature[self.intervals[i][j][2]][self.dest_num] = self.intervals[i][j][1] - t
                    self.network_feature[self.intervals[i][j][3]][self.dest_num] = self.intervals[i][j][1] - t

        self.adj = sparse.coo_matrix(self.adj)
        self.adj = SparseTensor(row=torch.LongTensor(self.adj.row), col=torch.LongTensor(self.adj.col))

    def simulate(self, args, input_cells):

        self.date_set = network_dataset(args, self.cell_list)
        input_indexs = [self.cell_index[cell+'-0'] for cell in input_cells]

        data = torch.Tensor(self.date_set[int(args["start"] / args["sim_step"])]).unsqueeze(0)
        batch = data.shape[0]
        time = data.shape[1]
        cell = data.shape[2]
        feature = self.network_feature.shape[1]
        net_feature = torch.Tensor(self.network_feature).expand(batch, time, cell, feature)

        cell_number = data.shape[2]
        dest_number = data.shape[3]

        assert cell_number == self.cell_num
        assert dest_number == len(self.destination)

        if args["use_cuda"]:
            self.simulator = self.simulator.cuda()
            data = data.cuda()
            net_feature = net_feature.cuda()

        data = torch.cat((data, net_feature), axis=3)

        outputs = data.data.new(1, args["temporal_length"] - args["init_length"], cell_number, dest_number).fill_(0).float()

        input_now = data[:, 0, :, :]

        for t in range(data.shape[1]-1):

            self.generate_adj(args["start"] + t*args["sim_step"])
            if args["use_cuda"]:
                self.adj = self.adj.cuda()
            output = self.simulator.infer(input_now, input_cells, self.adj)
            
            if t >= args["init_length"]:
                outputs[:, t-args["init_length"], :, :] += output
                input_now = torch.cat((output[:, :, :], net_feature[:, t, :, :]), axis=2)
                input_now[:, input_indexs, :] = data[:, t+1, input_indexs, :]
            else:
                input_now = data[:, t+1, :,:]

        return outputs

    def heat_map(self, output, cell_list):

        index = [self.cell_index[cell] for cell in cell_list]

        data = output[:, :, index, :].squeeze(0)
        data = data.cpu().numpy().sum(axis=2)

        sns.set()
        ax = sns.heatmap(data.T, fmt="d",cmap='YlGnBu')
        plt.show()

    def train(self, args, input_cells):

        with open(os.path.join(args["fold"], args["config"]+'.yaml')) as config:
            args = yaml.load(config)
        
        criterion = nn.MSELoss()

        if args["use_cuda"]:
            self.simulator = self.simulator.cuda()
            criterion = criterion.cuda()
            self.network_feature = self.network_feature.cuda()

        self.simulator.train()
        optimizer = torch.optim.Adagrad(self.simulator.parameters(), weight_decay=args["lambda_param"])
        acc_meter = meter.AverageValueMeter()
        last_frame_meter = meter.AverageValueMeter()

        best_acc_epoch = 0
        best_acc = float('inf')

        data_args = {}
        data_args["sim_step"] = args["sim_step"]
        data_args["delta_T"] = args["delta_T"]
        data_args["temporal_length"] = args["temporal_length"]
        data_args["data_fold"] = args["data_fold"]

        for epoch in range(args["num_epochs"]):

            print("***********************start training***************************")
            acc_meter.reset()
            start = time.time()

            for prefix in args["prefix"]:
                data_args["prefix"] = prefix
                data_set = network_dataset(args, self.cell_list)
                data_loader = DataLoader(data_set, batch_size=args["batch_size"])

                for i, data in tqdm(enumerate(data_loader)):

                    self.simulator.zero_grad()
                    optimizer.zero_grad()

                    data = Variable(data).float()

                    if args["use_cuda"]:
                        data = data.cuda()
                    
                    net_feauture = self.network_feature.expand(data.shape)
                    data = torch.cat((data, net_feauture), axis = 3)


                        



if __name__ == "__main__":
    
    with open("test_net.pkl", 'rb') as f:
        net_information = pickle.load(f)
    destiantion = [end+'-2' for end in utils.end_edge.keys()]
    args = {}
    args["sim_step"] = 0.1
    args["delta_T"] = 5
    args["temporal_length"] = 80
    args["init_length"] = 40
    args["prefix"] = "default"
    args["data_fold"] = "data"
    args["start"] = 0
    args["use_cuda"] = True
    args["dest_number"] = 6
    start_cell = [cell for cell in utils.start_edge.keys()]
    model = test_model(args)
    a = network(net_information, model, destiantion)
    output = a.simulate(args, start_cell)
    show_cell = ["gneE0-0", "gneE0-1", "gneE0-2"]
    a.heat_map(output, show_cell)
