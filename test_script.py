from test import test_model
import yaml
import os
import numpy as np

import dir_manage as d

basic_conf = os.path.join(d.config_data_path, "four_large_test.yaml")

with open(basic_conf, 'rb') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

model_dict = {"st_node_encoder":"one_hot_longer", "non_dir_model":"non_dir"}
data_prefix = ["four_low3", "four_low5", "four_6", "four_5"]
model_result = {"st_node_encoder": np.zeros((30, 4)), "non_dir_model": np.zeros((30, 4))}
model_range = list(range(20, 50))
args["plot"] = False

for model_type in model_dict.keys():
    model_prefix = model_dict[model_type]
    args["model_type"] = model_type
    args["model_prefix"] = model_prefix
    args["show_cell"] = None
    for i in model_range:
        args["model"] = i
        for j in range(len(data_prefix)):
            args["prefix"] = [data_prefix[j]]
            model_result[model_type][i-20, j] = test_model(args)

print(model_result["st_node_encoder"])
print(model_result["non_dir_model"])

np.save("st_node_encoder.npy", model_result["st_node_encoder"])
np.save("non_dir_model", model_result["non_dir_model"])
