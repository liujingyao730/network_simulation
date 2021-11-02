import numpy as np
import os
from numpy.core.fromnumeric import var
from numpy.lib.function_base import select
import pandas as pd
import pickle
import geatpy as ea
import time

import dir_manage
from S_model import network

mean_value = True

net_pickle = "four_large.pkl"
inputs_file = os.path.join(dir_manage.cell_data_path, "four_2_inputs.csv")
if mean_value:
    target_file = os.path.join(dir_manage.cell_data_path, "four_2_target_mean.csv")
else:
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

# if mean_value:
#     i = 0
#     a = pd.DataFrame(columns=target.columns)
#     while i+900 < target.shape[0]:
#         start_time = target.index[i]
#         end_time = target.index[i+899]
#         a.loc[i] = target.loc[start_time:end_time].sum() / 900
#         i += 1
#     target = a

inputs = inputs.loc[inputs_time_list]
target = target.loc[outputs_time_list]

def loss_func(Phen):

    global link_number, inputs, target, net

    popluation_size, length = Phen.shape
    result = np.zeros((popluation_size, 1))

    assert length == 7 * link_number

    for i in range(popluation_size):

        x = Phen[i, :]
        net.staturated_flow = x[:3*link_number].reshape((link_number, 3))
        net.split_rate = x[3*link_number:6*link_number].reshape((link_number, 3))
        net.capacity[:-1] = x[6*link_number:]

        result[i, 0] += net.calculate_loss(inputs, target, input_links, output_links)
    
    return result
'''
def loss_func(Phen):

    population, length = Phen.shape

    Phen = Phen * Phen
    return np.sum(Phen, axis=1).reshape(population, 1)
'''
var_number = 7 * link_number

upper = np.ones(var_number)
lower = np.zeros(var_number)
lower[:3*link_number] = 1
upper[:3*link_number] = 5
lower[6*link_number:] = 100
upper[6*link_number:] = 2500
ranges = np.vstack([lower, upper])

is_border = np.ones((2, var_number))

varTypes = np.zeros(var_number)

Encoding = "BG"
coders = [1 for i in range(var_number)]
precisions = [4 for i in range(var_number)]
scales = [0 for i in range(var_number)]

FieldD = ea.crtfld(Encoding, varTypes, ranges, is_border, precisions, coders, scales)

NIND = 200
MAXGEN = 2000
maxormins = np.array([1])

selectStyle = "sus"
recStyle = "xovdp"
mutStyle = "mutbin"
Lind = int(np.sum(FieldD[0, :]))

pc = 0.9
pm = 1 / Lind
obj_trace = np.zeros((MAXGEN, 2))
var_trace = np.zeros((MAXGEN, Lind))

start_time = time.time()
Chorm = ea.crtpc(Encoding, NIND, FieldD)
variable = ea.bs2ri(Chorm, FieldD)
ObjV = loss_func(variable)
best_ind = np.argmin(ObjV)

for gen in range(MAXGEN):
    
    FitnV = ea.ranking(ObjV, maxormins=maxormins)
    SelCh = Chorm[ea.selecting(selectStyle, FitnV, NIND-1), :]
    SelCh = ea.recombin(recStyle, SelCh, pc)
    SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)

    Chorm = np.vstack([Chorm[best_ind, :], SelCh])
    Phen = ea.bs2ri(Chorm, FieldD)
    ObjV = loss_func(Phen)

    best_ind = np.argmin(ObjV)
    obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]
    obj_trace[gen, 1] = ObjV[best_ind]
    var_trace[gen, :] = Chorm[best_ind, :]
    print(gen, ObjV[best_ind])

end_time = time.time()
# ea.trcplot(obj_trace, [['种群个体平均目标函数值','种群最优个体目标函数值']])

best_gen = np.argmin(obj_trace[:, [1]])
print("得到的最优函数值为 ", obj_trace[best_gen, 1])
variable = ea.bs2ri(var_trace[[best_gen], :], FieldD)
np.save("genetic_solution.npy", variable)
# for i in range(variable.shape[1]):
#     print('x'+str(i)+'=', variable[0, i])
print("用时 ", end_time-start_time, "s")
