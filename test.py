import torch
import pickle
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

from model import GCN_GRU
from network import network_data
import dir_manage as d
from utils import sparselist_to_tensor

basic_conf = os.path.join(d.config_data_path, "default.yaml")

with open(basic_conf, 'rb') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

args["prefix"] = "two_6"

args["init_length"] = 4
args["temporal_length"] = 1400
args["batch_size"] = 1
args["net_file"] = "two_net.pkl"
args["model_prefix"] = "only_two_GRU_GCN"
args["model"] = "25"

with open(os.path.join(d.cell_data_path, args["net_file"]), 'rb') as f:
    net_information = pickle.load(f)

data_set = network_data(net_information, args["destination"], args["prefix"], args)

inputs, target, adj_list = data_set.get_batch()

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

f = torch.nn.MSELoss()
output = torch.sum(output, dim=3)
target = torch.sum(target[:, :, :, :args["output_size"]], dim=3)

print(f(output, target))
print(f(output[:, -1, :], target[:, -1, :]))

real_cell = target[0, :, 0].detach().cpu().numpy()
predict_cell = output[0, :, 0].detach().cpu().numpy()
x = np.array(range(100))

plt.figure()
plt.plot(x, real_cell[:100], label="gt")
plt.plot(x, predict_cell[:100], label="pd")
plt.legend()
plt.savefig("123.png")
