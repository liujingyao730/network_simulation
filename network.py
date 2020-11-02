import pandas as pd
import os
import numpy as np
import networkx as nx
from scipy import sparse
import pickle
import pylab
import torch

import route_conf
import dir_manage as d

class network_data(object):
    def __init__(self, net_information, destination, prefix, args):

        self.ordinary_cell = net_information["ordinary_cell"]
        self.junction_cell = net_information["junction_cell"]
        self.connection = net_information["connection"]
        self.signal_connection = net_information["signal_connection"]
        self.tlc_time = net_information["tlc_time"]

        self.destination = destination
        self.prefix = prefix

        self.data_fold = d.cell_data_path
        self.sim_step = args.get("sim_step", 0.1)
        self.batch_size = args.get("batch_size", None)
        self.init_length = args.get("init_length", 4)
        self.temporal_length = args.get("temporal_length", 8)
        self.deltaT = args.get("deltaT", 5)
        self.step = int(self.deltaT / self.sim_step)

        self.dest_size = len(self.destination)
        self.input_size = self.dest_size + 1  # 目的地数目加上通行时间间隔

        self.cell_list = []
        for edge in self.ordinary_cell:
            self.cell_list.extend(self.ordinary_cell[edge]["cell_id"])

        self.N = len(self.cell_list)
        self.cell_index = {self.cell_list[i]: i for i in range(self.N)}
        self.dest_index = {
            self.destination[i]: i
            for i in range(self.dest_size)
        }

        self.generate_base_adj()
        self.generate_interval()
        self.generate_network_feature()
        self.generate_all_adj()
        self.load_cell_data()

    def generate_base_adj(self):

        self.rows = [self.cell_index[cell] for cell in self.connection]
        self.cols = [
            self.cell_index[self.connection[cell]]
            for cell in self.connection.keys()
        ]
        self.vals = [1 for cell in self.connection.keys()]

        self.rows.extend([i for i in range(self.N)])
        self.cols.extend([i for i in range(self.N)])
        self.vals.extend([1 for i in range(self.N)])

        self.base_adj = sparse.csc_matrix((self.vals, (self.rows, self.cols)),
                                          shape=(self.N, self.N))

    def generate_interval(self):

        self.intervals = [[] for i in range(len(self.tlc_time))]
        self.loc_adj = [[] for i in range(len(self.tlc_time))]

        for from_cell in self.signal_connection.keys():
            for to_cell in self.signal_connection[from_cell].keys():
                junction_id = self.signal_connection[from_cell][to_cell][2]
                from_id = self.cell_index[from_cell]
                to_id = self.cell_index[to_cell]
                start = self.signal_connection[from_cell][to_cell][0]
                end = self.signal_connection[from_cell][to_cell][1]
                self.intervals[junction_id].append(
                    [start, end, from_id, to_id])
                self.loc_adj[junction_id].append(
                    sparse.csc_matrix(([1], ([from_id], [to_id])),
                                      shape=(self.N, self.N)))
                self.rows.append(from_id)
                self.cols.append(to_id)
                self.vals.append(1)

        self.all_adj = sparse.csc_matrix((self.vals, (self.rows, self.cols)),
                                         shape=(self.N, self.N))

    def generate_network_feature(self):

        row = self.rows[:]
        col = self.cols[:]

        edges = [(row[i], col[i]) for i in range(len(row))]

        G = nx.DiGraph()
        N = nx.path_graph(self.N)
        G.add_nodes_from(N)
        G.add_edges_from(edges)

        self.network_feature = np.zeros((self.N, self.input_size)) - 1
        self.network_feature[:, -1] = 100

        for i in range(self.N):
            for dest in self.destination:
                j = self.dest_index[dest]
                try:
                    self.network_feature[i][j] += nx.dijkstra_path_length(
                        G, source=i, target=j)
                except nx.exception.NetworkXNoPath:
                    pass
        '''
        pos = nx.spring_layout(G)
        nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5)
        pylab.title('topology',fontsize=15)
        pylab.savefig("graph.png")
        '''

    def generate_all_adj(self):

        cycle = [int(tlc[0]) for tlc in self.tlc_time]
        offset = [tlc[1] for tlc in self.tlc_time]
        self.cycle_lcm = np.lcm.reduce(cycle)
        self.adj_with_time = []

        for index in range(int(self.cycle_lcm / self.sim_step)):
            base_time = index * self.sim_step
            self.adj_with_time.append(self.base_adj)
            for junction_id in range(len(offset)):
                time = (base_time + offset[junction_id]) % cycle[junction_id]
                for connect_id in range(len(self.intervals[junction_id])):
                    if self.intervals[junction_id][connect_id][
                            0] <= time <= self.intervals[junction_id][
                                connect_id][1]:
                        self.adj_with_time[index] += self.loc_adj[junction_id][
                            connect_id]
                        self.network_feature[self.intervals[junction_id][
                            connect_id][2]][self.dest_size] = self.intervals[
                                junction_id][connect_id][1] - time

        # np.save(os.path.join(self.data_fold, self.prefix+'.npy'), self.adj_with_time)

    def load_cell_data(self):

        dest1_file = os.path.join(self.data_fold, self.prefix + "_dest1.csv")
        dest2_file = os.path.join(self.data_fold, self.prefix + "_dest2.csv")
        dest3_file = os.path.join(self.data_fold, self.prefix + "_dest3.csv")
        dest4_file = os.path.join(self.data_fold, self.prefix + "_dest4.csv")
        dest5_file = os.path.join(self.data_fold, self.prefix + "_dest5.csv")
        dest6_file = os.path.join(self.data_fold, self.prefix + "_dest6.csv")

        dest1 = pd.read_csv(dest1_file,
                            index_col=0)[self.cell_list].values[:, :, None]
        dest2 = pd.read_csv(dest2_file,
                            index_col=0)[self.cell_list].values[:, :, None]
        dest3 = pd.read_csv(dest3_file,
                            index_col=0)[self.cell_list].values[:, :, None]
        dest4 = pd.read_csv(dest4_file,
                            index_col=0)[self.cell_list].values[:, :, None]
        dest5 = pd.read_csv(dest5_file,
                            index_col=0)[self.cell_list].values[:, :, None]
        dest6 = pd.read_csv(dest6_file,
                            index_col=0)[self.cell_list].values[:, :, None]
        network_feature = np.tile(self.network_feature, (dest1.shape[0], 1, 1))

        self.data = np.concatenate(
            (dest1, dest2, dest3, dest4, dest5, dest6, network_feature),
            axis=2)
        self.time_bound = self.data.shape[0] - self.step * self.temporal_length

        self.cycle_step = int(self.cycle_lcm / self.sim_step)
        self.max_batch_size = int(self.data.shape[0] / self.cycle_step)

        if self.batch_size is None:
            self.batch_size = self.max_batch_size
        self.index = 0

    def next_index(self):

        self.index += 1

        phase = self.index % self.cycle_step
        if phase == 0:
            self.index += self.cycle_step * self.batch_size

        if self.index > self.time_bound:
            return False

        return True

    def get_item(self, point):

        input_time = [
            point + i * self.step for i in range(self.temporal_length)
        ]
        output_time = [
            point + (i + self.init_length+1) * self.step
            for i in range(self.temporal_length - self.init_length)
        ]

        input_data = self.data[input_time, :, :]
        output_data = self.data[output_time, :, :]

        return (input_data, output_data)

    def get_adj_list(self, point):

        adj_list = []

        for i in range(self.temporal_length):
            adj = self.adj_with_time[(point + i * self.step) % self.cycle_lcm]
            adj_list.append(adj)

        return adj_list

    def get_batch(self):

        input_data, output_data = self.get_item(self.index)
        input_datas = input_data[None, :, :, :]
        output_datas = output_data[None, :, :, :]

        for i in range(self.batch_size - 1):
            point = (1 + i) * self.cycle_step
            if point > self.time_bound:
                break
            input_data, output_data = self.get_item(point)
            input_datas = np.concatenate(
                (input_datas, input_data[None, :, :, :]), axis=0)
            output_datas = np.concatenate(
                (output_datas, output_data[None, :, :, :]), axis=0)

        adj_list = self.get_adj_list(self.index)

        return input_datas, output_datas, adj_list
    
    def normalize_data(self):

        self.mean = np.mean(self.data[:, :, :self.dest_size], axis=(0, 1))
        self.std = np.std(self.data[:, :, :self.dest_size], axis=(0, 1))

        self.std[self.std == 0] = 1

        self.data[:, :, :self.dest_size] -= self.mean[None, None, :]
        self.data[:, :, :self.dest_size] /= self.std[None, None, :]

    def recovery_data(self, data):

        [batch_size, temporal, cells, feature] = data.shape

        assert feature >= self.dest_size

        if isinstance(data, np.ndarray):

            data[:, :, :, :self.dest_size] *= self.std[None, None, None, :]
            data[:, :, :, :self.dest_size] += self.mean[None, None, None, :]
        
        else:

            tmp_std = data.data.new(1, 1, 1, self.dest_size).fill_(torch.Tensor(self.std[None, None, None, :])).float()
            tmp_mean = data.data.new(1, 1, 1, self.dest_size).fill_(torch.Tensor(self.mean[None, None, None, :])).float()

            data[:, :, :, :self.dest_size] *= tmp_std
            data[:, :, :, :self.dest_size] += tmp_mean

        return data
    
    def reset_index(self):

        self.index = 0
    
    def name_to_id(self, names):

        return [self.cell_index[name] for name in names]


if __name__ == "__main__":

    with open("data/input_data/test_net.pkl", 'rb') as f:
        net_information = pickle.load(f)
    destiantion = [end + '-2' for end in route_conf.end_edge.keys()]
    prefix = "default"
    args = {}
    args["sim_step"] = 0.1
    args["deltaT"] = 5
    args["temporal_length"] = 8
    args["init_length"] = 4
    args["data_fold"] = "data"
    args["start"] = 0
    args["use_cuda"] = True
    args["dest_number"] = 6
    start_cell = [cell for cell in route_conf.start_edge.keys()]
    a = network_data(net_information, destiantion, prefix, args)
    a.normalize_data()
    inputs, outputs, adj_list = a.get_batch()
    input = a.recovery_data(inputs)
    b = 1
