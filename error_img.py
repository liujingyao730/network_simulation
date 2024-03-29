import torch
import pickle
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import imageio

from model import GCN_GRU, node_encode_attention
from coder_model import st_node_encoder
from network import network_data, data_on_network
import dir_manage as d
from utils import sparselist_to_tensor
import pre_process as pp

basic_conf = os.path.join(d.config_data_path, "four_large_test.yaml")

with open(basic_conf, 'rb') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

net_type = pp.calculate_layout("four_large.net.xml", pickle_file="test.pkl", best_distance=50)
figure_size = (15, 10)

show_time = 100

with open(os.path.join(d.cell_data_path, args["net_file"]), 'rb') as f:
    net_information = pickle.load(f)

data_set = data_on_network(net_information, args["destination"][0], args["prefix"], args)

inputs, adj_list = data_set.get_batch()
target = inputs[:, args["init_length"]+1:, :, :]

origin_adj = adj_list.copy()

# model = GCN_GRU(args)
# model = node_encode_attention(args)
model = st_node_encoder(args)
model_file = os.path.join(d.log_path, args["model_prefix"], str(args["model"])+'.tar')
checkpoint = torch.load(model_file)
model.load_state_dict(checkpoint["state_dict"])

inputs = torch.Tensor(inputs)
target = torch.Tensor(target)
adj_list = torch.Tensor(sparselist_to_tensor(adj_list))

inputs = inputs.cuda()
target = target.cuda()
adj_list = adj_list.cuda()
model = model.cuda()
cell_index = data_set.name_to_id(args["input_cells_name"])
model.set_input_cells(cell_index)

output = model.infer(inputs, adj_list)

output = torch.sum(output, dim=3).detach().cpu().numpy()[0, :, :]
target = torch.sum(target[:, :, :, :args["output_size"]], dim=3).detach().cpu().numpy()[0, :, :]

target = np.around(target, decimals=2)
output = np.around(output, decimals=2)

file_list = []
for i in range(show_time):
    file_list.append(os.path.join(d.pic_path, str(i)+'.png'))
    data_set.show_adj(origin_adj[i+args["init_length"]], file=file_list[i], colors=target[i, :], with_label=True, network_type=net_type, figure_size=figure_size)

frames = []
for img_name in file_list:
    frames.append(imageio.imread(img_name))

imageio.mimsave("targets.gif", frames, 'GIF', duration=0.2)

file_list = []
for i in range(show_time):
    file_list.append(os.path.join(d.pic_path, str(i)+'_out.png'))
    data_set.show_adj(origin_adj[i+args["init_length"]], file=file_list[i], colors=output[i, :], with_label=True, network_type=net_type, figure_size=figure_size)

frames = []
for img_name in file_list:
    frames.append(imageio.imread(img_name))

imageio.mimsave("outputs.gif", frames, 'GIF', duration=0.2)
'''
file_list = []
for i in range(show_time):
    file_list.append(os.path.join(d.pic_path, str(i)+'.png'))
    data_set.show_adj(origin_adj[i-args["init_length"]], file=file_list[i], colors=np.round(target[i, :] - output[i, :], decimals=2), with_label=True, network_type=net_type)

frames = []
for img_name in file_list:
    frames.append(imageio.imread(img_name))

imageio.mimsave("errors.gif", frames, 'GIF', duration=0.2)
'''
