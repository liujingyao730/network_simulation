import torch
import pickle
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

from model import GCN_GRU, node_encode_attention, node_encode_attention_res
from coder_model import st_node_encoder, st_node_encoder_res
from network import network_data, data_on_network
import dir_manage as d
from utils import sparselist_to_tensor
import pre_process as pp

basic_conf = os.path.join(d.config_data_path, "four_test.yaml")
show_detail = False

with open(basic_conf, 'rb') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

with open(os.path.join(d.cell_data_path, args["net_file"]), 'rb') as f:
    net_information = pickle.load(f)

data_set = data_on_network(net_information, args["destination"][0], args["prefix"], args)
# data_set.normalize_data()

inputs, adj_list = data_set.get_batch()
target = inputs[:, args["init_length"]+1:, :, :]

'''
net_type = pp.calculate_layout("four.net.xml", "test.pkl")
for i in range(len(adj_list)):
    print(i)
    data_set.show_adj(adj_list[i], network_type=net_type, with_label=True, colors=target[0, 0, :, -2])
'''
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

# target = data_set.recovery_data(target)
# output = data_set.recovery_data(output)

if show_detail:
    
    real_data = target[0, :, :, :].detach().cpu().numpy()
    predict_data = output[0, :, :, :].detach().cpu().numpy()

    for i in range(8):
        
        real_cell = real_data[:200, 32, i]
        predict_cell = predict_data[:200, 32, i]

        x = np.array(range(real_cell.shape[0]))

        plt.figure()
        plt.plot(x, real_cell, label="gt")
        plt.plot(x, predict_cell, label="pd")
        plt.legend()
        plt.savefig("dest" + str(i) + ".png")        

f = torch.nn.MSELoss()
output = torch.sum(output, dim=3)
target = torch.sum(target[:, :, :, :args["output_size"]], dim=3)

max_error = 0
max_error_cell = -1

for i in range(output.shape[2]):
    error = f(output[0, :, i], target[0, :, i])
    if error > 25:
        print(i, error)
    if error > max_error:
        max_error = error
        max_error_cell = i

print("max error ", max_error, " in cell ", max_error_cell)
print(f(output, target))
print(f(output[:, -1, :], target[:, -1, :]))

real_cell = target[0, :, 20].detach().cpu().numpy()
predict_cell = output[0, :, 20].detach().cpu().numpy()
x = np.array(range(real_cell.shape[0]))

plt.figure()
plt.plot(x, real_cell, label="gt")
plt.plot(x, predict_cell, label="pd")
plt.legend()
plt.savefig("123.png")
