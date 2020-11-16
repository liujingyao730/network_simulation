import torch
import pickle
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import imageio

from model import GCN_GRU
from network import network_data
import dir_manage as d
from utils import sparselist_to_tensor

basic_conf = os.path.join(d.config_data_path, "default.yaml")

with open(basic_conf, 'rb') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

args["prefix"] = "two_6"

args["init_length"] = 4
args["temporal_length"] = 60
args["batch_size"] = 1
args["net_file"] = "two_net.pkl"
args["model_prefix"] = "GRU_GCN"
args["model"] = "9"

show_time = 50

with open(os.path.join(d.cell_data_path, args["net_file"]), 'rb') as f:
    net_information = pickle.load(f)

data_set = network_data(net_information, args["destination"], args["prefix"], args)
data_set.normalize_data()

inputs, target, adj_list = data_set.get_batch()

origin_adj = adj_list.copy()

model = GCN_GRU(args)
model_file = os.path.join(d.log_path, args["model_prefix"], args["model"]+'.tar')
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

target = data_set.recovery_data(target)
output = data_set.recovery_data(output)

output = torch.sum(output, dim=3).detach().cpu().numpy()[0, :, :]
target = torch.sum(target[:, :, :, :args["output_size"]], dim=3).detach().cpu().numpy()[0, :, :]

output = np.around(output, decimals=2)

file_list = []
for i in range(show_time):
    file_list.append(os.path.join(d.pic_path, str(i)+'.png'))
    data_set.show_adj(origin_adj[i], file=file_list[i], colors=target[i, :], with_label=True)

frames = []
for img_name in file_list:
    frames.append(imageio.imread(img_name))

imageio.mimsave("targets.gif", frames, 'GIF', duration=0.2)

file_list = []
for i in range(show_time):
    file_list.append(os.path.join(d.pic_path, str(i)+'_out.png'))
    data_set.show_adj(origin_adj[i], file=file_list[i], colors=output[i, :], with_label=True)

frames = []
for img_name in file_list:
    frames.append(imageio.imread(img_name))

imageio.mimsave("outputs.gif", frames, 'GIF', duration=0.2)

file_list = []
for i in range(show_time):
    file_list.append(os.path.join(d.pic_path, str(i)+'_error.png'))
    data_set.show_adj(origin_adj[i], file=file_list[i], colors=np.round(target[i, :] - output[i, :], decimals=2), with_label=True)

frames = []
for img_name in file_list:
    frames.append(imageio.imread(img_name))

imageio.mimsave("errors.gif", frames, 'GIF', duration=0.2)
