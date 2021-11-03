from test import test_model
import yaml
import os
import numpy as np
import pickle

import dir_manage as d
from network import data_on_network

basic_conf = os.path.join(d.config_data_path, "four_large_test.yaml")

with open(basic_conf, 'rb') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

# model_dict = {"replaceable_model": "fix_42_2"}
# model_dict = {"dyn_embedding": "dyn_embedding_256"}
# model_dict = {"single_attention":"single_gate_2_8"}
# model_dict = {"baseline": "gat_lstm_1_4"}
model_dict = {"GMAN":"gman"}
data_prefix = ["four_low3", "four_low5", "four_6", "four_5"]
data_sets = {}
# model_result = {"replaceable_model": np.zeros((30, 4))}
# model_result = {"single_attention": np.zeros((30, 4))}
# model_result = {"baseline": np.zeros((60, 4))}
model_result = {"GMAN": np.zeros((30, 4))}
model_range = list(range(20, 50))
args["plot"] = False
with open(os.path.join(d.cell_data_path, args["net_file"]), 'rb') as f:
    net_information = pickle.load(f)
args["temporal_length"] = 750
args["temporal_extend"] = 0
for prefix in data_prefix:
    data_sets[prefix] = data_on_network(net_information, args["destination"][0], [prefix], args)

for model_type in model_dict.keys():
    model_prefix = model_dict[model_type]
    args["model_type"] = model_type
    args["model_prefix"] = model_prefix
    args["show_cell"] = None
    args["hidden_size"] = 256
    args["gnn"] = "gcn"
    args["rnn"] = "lstm"
    args["plot"] = False
    args["get_eva_time"] = False
    args["show_detail"] = False
    for i in model_range:
        args["model"] = i
        print(i)
        for j in range(len(data_prefix)):
            args["prefix"] = [data_prefix[j]]
            model_result[model_type][i-20, j] = test_model(args, data_sets[data_prefix[j]])

# print(model_result["replaceable_model"])
# print(model_result["dyn_embedding"])
# print(model_result["single_attention"])
# print(model_result["baseline"])

# np.save("weight_42_2.npy", model_result["replaceable_model"])
# np.save("dyn_embedding_256.npy", model_result["dyn_embedding"])
# np.save("single_att_2_8.npy", model_result["single_attention"])
# np.save("gat_lstm.npy", model_result["baseline"])
np.save("gman.npy", model_result["GMAN"])
