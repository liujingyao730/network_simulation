import pickle
import networkx as nx
import os
from scipy import sparse
from torch._C import layout
import yaml
import matplotlib.pyplot as plt

import dir_manage as d
from generateSE import learn_embeddings
from network import data_on_network
import node2vec

is_directed = True
p = 2
q = 1
num_walks = 100
walk_length = 80
dimensions = 64
iter = 1000
SE_file = os.path.join(d.cell_data_path, "four_large_SE.txt")

net_file = os.path.join(d.cell_data_path, 'four_large.pkl')
with open(net_file, 'rb') as f:
    net_information = pickle.load(f)

with open(os.path.join(d.config_data_path, "four_large_train.yaml"), 'rb') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

destination = args["destination"][0]
prefix_list = ['four_low1']

data_set = data_on_network(net_information, destination, prefix_list, args)
adj = sparse.coo_matrix(data_set.all_adj)
row = adj.row
col = adj.col
edges = [(row[i], col[i]) for i in range(len(row))]

nx_G = nx.DiGraph()
N = nx.path_graph(data_set.N)
nx_G.add_nodes_from(N)
for edge in edges:
    nx_G.add_edge(edge[0], edge[1], weight=1)
# nx_G.add_edges_from(edges)
nx.draw(nx_G, pos=nx.spring_layout(nx_G), font_size=5, with_labels=True, node_size=15)
plt.savefig('graph.png')
G = node2vec.Graph(nx_G, is_directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)
learn_embeddings(walks, dimensions, SE_file)
