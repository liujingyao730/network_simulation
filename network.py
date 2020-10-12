import pandas as pd
import os 
import numpy as np
import networkx as nx
from scipy import sparse
import pickle
import pylab
import matplotlib.pyplot as plt

import utils

class network_data(object):

    def __init__(self, net_information, destination, args):
        
        self.ordinary_cell = net_information["ordinary_cell"]
        self.junction_cell = net_information["junction_cell"]
        self.connection = net_information["connection"]
        self.signal_connection = net_information["signal_connection"]
        self.tlc_time = net_information["tlc_time"]

        self.destination = destination

        self.prefix = args.get("prefix", "default")
        self.data_fold = args.get("data_fold", "data")
        self.sim_step = args.get("sim_step", 0.1)

        self.dest_size = len(self.destination)
        self.input_size = self.dest_size + 1  # 目的地数目加上通行时间间隔

        self.cell_list = []
        for edge in self.ordinary_cell:
            self.cell_list.extend(self.ordinary_cell[edge]["cell_id"])
        
        self.N = len(self.cell_list)
        self.cell_index = {self.cell_list[i]:i for i in range(self.N)}
        self.dest_index = {self.destination[i]:i for i in range(self.dest_size)}

        self.generate_base_adj()
        self.generate_interval()
        self.generate_network_feature()
        self.generate_all_adj(self.sim_step)

    def generate_base_adj(self):

        self.rows = [self.cell_index[cell] for cell in self.connection]
        self.cols = [self.cell_index[self.connection[cell]] for cell in self.connection.keys()]
        self.vals = [1 for cell in self.connection.keys()]
        
        self.rows.extend([i for i in range(self.N)])
        self.cols.extend([i for i in range(self.N)])
        self.vals.extend([1 for i in range(self.N)])

        self.base_adj = sparse.csc_matrix((self.vals, (self.rows, self.cols)), shape=(self.N, self.N))

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
                self.intervals[junction_id].append([start, end, from_id, to_id])
                self.loc_adj[junction_id].append(
                    sparse.csc_matrix(([1], ([from_id], [to_id])), shape=(self.N, self.N))
                )
                self.rows.append(from_id)
                self.cols.append(to_id)
                self.vals.append(1)

        self.all_adj = sparse.csc_matrix((self.vals, (self.rows, self.cols)), shape=(self.N, self.N))
    
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
                    self.network_feature[i][j] += nx.dijkstra_path_length(G, source=i, target=j)
                except nx.exception.NetworkXNoPath:
                    pass
        
        pos = nx.spring_layout(G)
        nx.draw(G,pos,with_labels=True, node_color='white', edge_color='red', node_size=400, alpha=0.5)
        pylab.title('topology',fontsize=15)
        pylab.savefig("graph.png")
    
    def generate_all_adj(self, sim_step):
        
        cycle = [int(tlc[0]) for tlc in self.tlc_time]
        offset = [tlc[1] for tlc in self.tlc_time]
        cycle_lcm = np.lcm.reduce(cycle)
        self.adj_with_time = []

        for index in range(int(cycle_lcm/sim_step)):
            base_time = index * sim_step
            self.adj_with_time.append(self.base_adj)
            for junction_id in range(len(offset)):
                time = (base_time + offset[junction_id]) % cycle[junction_id]
                for connect_id in range(len(self.intervals[junction_id])):
                    if self.intervals[junction_id][connect_id][0] <= time <= self.intervals[junction_id][connect_id][1]:
                        self.adj_with_time[index] += self.loc_adj[junction_id][connect_id]
                        self.network_feature[self.intervals[junction_id][connect_id][2]][self.dest_size] = self.intervals[junction_id][connect_id][1] - time

        np.save(os.path.join(self.data_fold, self.prefix+'.npy'), self.adj_with_time)

    def load_cell_data(self, file, args):

        self.data = pd.read_csv(file, index_col=0)[self.cell_list].value
    
    def __getitem__(self, index):

        return None
    
    def __len__(self):

        return 10


if __name__ == "__main__":
    
    with open("test_net.pkl", 'rb') as f:
        net_information = pickle.load(f)
    destiantion = [end + '-2' for end in utils.end_edge.keys()]
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
    a = network_data(net_information, destiantion, args)
