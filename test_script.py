from test import test_model
import yaml
import os
import numpy as np

import dir_manage as d

basic_conf = os.path.join(d.config_data_path, "four_large_test.yaml")

with open(basic_conf, 'rb') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

# model_dict = {"replaceable_model": "change_weight_2"}
# model_dict = {"dyn_embedding": "dyn_embedding_256"}
model_dict = {"single_attention":"single_gate3"}
data_prefix = ["four_low3", "four_low5", "four_6", "four_5"]
# model_result = {"replaceable_model": np.zeros((30, 4))}
model_result = {"single_attention": np.zeros((30, 4))}
model_range = list(range(20, 50))
args["plot"] = False

for model_type in model_dict.keys():
    model_prefix = model_dict[model_type]
    args["model_type"] = model_type
    args["model_prefix"] = model_prefix
    args["show_cell"] = None
    args["hidden_size"] = 256
    args["gnn"] = "gcn"
    args["rnn"] = "lstm"
    for i in model_range:
        args["model"] = i
        print(i)
        for j in range(len(data_prefix)):
            args["prefix"] = [data_prefix[j]]
            model_result[model_type][i-20, j] = test_model(args)

# print(model_result["replaceable_model"])
# print(model_result["dyn_embedding"])
print(model_result["single_attention"])

# np.save("change_weight_2.npy", model_result["replaceable_model"])
# np.save("dyn_embedding_256.npy", model_result["dyn_embedding"])
np.save("single_att3.npy", model_result["single_attention"])
