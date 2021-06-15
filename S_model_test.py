import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import dir_manage
from S_model import network

net_pickle = "four_large.pkl"
inputs_file = os.path.join(dir_manage.cell_data_path, "four_2_inputs.csv")
target_file = os.path.join(dir_manage.cell_data_path, "four_2_target.csv")
inputs = pd.read_csv(inputs_file, index_col=0)
target = pd.read_csv(target_file, index_col=0)
with open(os.path.join(dir_manage.cell_data_path, net_pickle), "rb") as f:
    net_information = pickle.load(f)
link_number = len(net_information["ordinary_cell"].keys())
inputs_time_list = [i*90 for i in range(40)]
outputs_time_list = [(i+1)*90 for i in range(40)]
net = network(net_pickle)
input_links = list(inputs.columns)
output_links = ["-gneE0", "-gneE1", "-gneE2", "-gneE4", "-gneE5", "-gneE6", "-gneE12", "-gneE13", "-gneE15", "-gneE14", "-gneE8", "-gneE10"]
inputs = inputs.loc[inputs_time_list]
target = target.loc[outputs_time_list]

x = np.load("four2.npy").flatten()

net.staturated_flow = x[:3*link_number].reshape((link_number, 3))
net.split_rate = x[3*link_number:6*link_number].reshape((link_number, 3))
net.capacity[:-1] = x[6*link_number]

output = net.calculate_loss(inputs, target, input_links, output_links, show_Detail=True)

fig = plt.figure()
x = range(output.shape[0])
plt.plot(x, output[:, 16], label="output")
plt.plot(x, target.values[:, 16], label="targets")
plt.legend()
plt.savefig("s_model.png")
