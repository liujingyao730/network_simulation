import numpy as np
import os
from numpy.lib.function_base import select
import pandas as pd
import pickle
import geatpy as ea
import time

import dir_manage
from S_model import network

net_pickle = "four_large.pkl"
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

def split_rate_loss(Phen):

    global link_number, inputs, target, net

    popluation_size, length = Phen.shape
    result = np.zeros((popluation_size, 1))

    for i in range(popluation_size):

        net.split_rate = Phen[i, :].reshape((link_number, 3))

        result[i, 0] += net.calculate_loss(inputs, target, input_links, output_links)

    return result

def capacity_loss(Phen):

    global link_number, inputs, target, net

    popluation_size, length = Phen.shape
    result = np.zeros((popluation_size, 1))

    for i in range(popluation_size):

        net.capacity[:-1] = Phen[i, :]
        result[i, 0] += net.calculate_loss(inputs, target, input_links, output_links)
    
    return result

def staturated_flow_loss(Phen):

    global link_number, inputs, target, net

    popluation_size, length = Phen.shape
    result = np.zeros((popluation_size, 1))

    for i in range(popluation_size):

        net.staturated_flow = Phen[i, :].reshape((link_number, 3))
        result[i, 0] += net.calculate_loss(inputs, target, input_links, output_links)

    return result

def evolution():

    global link_number, net

    net.split_rate = np.ones((link_number, 3)) / 3
    net.capacity = np.ones(link_number+1) * 1000
    net.staturated_flow = np.ones((link_number, 3)) * 2.5

    split_var = 3 * link_number
    staturated_var = 3 * link_number
    capacity_var = link_number

    split_lower = np.zeros(split_var)
    staturated_lower = np.ones(staturated_var) * 0.5
    capacity_lower = np.ones(capacity_var) * 700
    split_upper = np.ones(split_var)
    staturated_upper = np.ones(staturated_var) * 4
    capacity_upper = np.ones(capacity_var) * 2500

    is_split_border = np.ones((2, split_var))
    is_staturated_border = np.ones((2, staturated_var))
    is_capacity_border = np.ones((2, capacity_var))
    split_varTypes = np.zeros(split_var)
    staturated_varTypes = np.zeros(staturated_var)
    capacity_varTypes = np.zeros(capacity_var)
    split_range = np.vstack([split_lower, split_upper])
    staturated_range = np.vstack([staturated_lower, staturated_upper])
    capacity_range = np.vstack([capacity_lower, capacity_upper])

    Encoding = "BG"
    split_coders = [1 for i in range(split_var)]
    staturated_coders = [1 for i in range(staturated_var)]
    capacity_coders = [1 for i in range(capacity_var)]
    split_precisions = [4 for i in range(split_var)]
    staturated_precisions = [4 for i in range(staturated_var)]
    capacity_precisions = [4 for i in range(capacity_var)]
    split_scales = [0 for i in range(split_var)]
    staturated_scales = [0 for i in range(staturated_var)]
    capacity_scales = [0 for i in range(capacity_var)]

    split_FieldD = ea.crtfld(Encoding, split_varTypes, split_range, is_split_border, split_precisions, split_coders, split_scales)
    staturated_FieldD = ea.crtfld(Encoding, staturated_varTypes, staturated_range, is_staturated_border, staturated_precisions, staturated_coders, staturated_scales)
    capacity_FieldD = ea.crtfld(Encoding, capacity_varTypes, capacity_range, is_capacity_border, capacity_precisions, capacity_coders, capacity_scales)

    NIND = 50
    MAXGEN = 100
    maxormins = np.array([1])

    selectStyle = "sus"
    recStyle = "xovdp"
    mutStyle = "mutbin"
    pc = 0.9
    split_Lind = int(np.sum(split_FieldD[0, :]))
    staturated_Lind = int(np.sum(staturated_FieldD[0, :]))
    capacity_Lind = int(np.sum(capacity_FieldD[0, :]))

    split_pm = 1 / split_Lind
    staturated_pm = 1 / staturated_Lind
    capacity_pm = 1 / capacity_Lind

    start_time = time.time()
    split_chorm = ea.crtpc(Encoding, NIND, split_FieldD)
    split_value = ea.bs2ri(split_chorm, split_FieldD)
    staturated_chorm = ea.crtpc(Encoding, NIND, staturated_FieldD)
    staturated_value = ea.bs2ri(staturated_chorm, staturated_FieldD)
    capacity_chorm = ea.crtpc(Encoding, NIND, capacity_FieldD)
    capacity_value = ea.bs2ri(capacity_chorm, capacity_FieldD)

    split_ObjV = split_rate_loss(split_value)
    staturated_ObjV = staturated_flow_loss(staturated_value)
    capacity_ObjV = capacity_loss(capacity_value)
    best_split_id = np.argmin(split_ObjV)
    best_staturated_id = np.argmin(staturated_ObjV)
    best_capacity_id = np.argmin(capacity_ObjV)

    net.split_rate = split_value[best_split_id, :].reshape((link_number, 3))
    net.staturated_flow = staturated_value[best_staturated_id, :].reshape((link_number, 3))
    net.capacity[:-1] = capacity_value[best_capacity_id, :]
    best_loss = float('inf')
    best_split = net.split_rate
    best_staturate = net.staturated_flow
    best_capacity = net.capacity[:-1]

    for gen in range(MAXGEN):

        FitnV = ea.ranking(split_ObjV, maxormins=maxormins)
        SelCh = split_chorm[ea.selecting(selectStyle, FitnV, NIND-1), :]
        SelCh = ea.recombin(recStyle, SelCh, pc)
        SelCh = ea.mutate(mutStyle, Encoding, SelCh, split_pm)

        split_chorm = np.vstack([split_chorm[best_split_id, :], SelCh])
        split_value = ea.bs2ri(split_chorm, split_FieldD)
        split_ObjV = split_rate_loss(split_value)
        best_split_id = np.argmin(split_ObjV)
        if split_ObjV[best_split_id] < best_loss:
            best_split = split_value[best_split_id]
            best_loss = split_ObjV[best_split_id]
            net.split_rate = best_split.reshape((link_number, 3))
        print("generation ", gen, " best split rate loss", split_ObjV[best_split_id])

        FitnV = ea.ranking(staturated_ObjV, maxormins=maxormins)
        SelCh = staturated_chorm[ea.selecting(selectStyle, FitnV, NIND-1), :]
        SelCh = ea.recombin(recStyle, SelCh, pc)
        SelCh = ea.mutate(mutStyle, Encoding, SelCh, staturated_pm)

        staturated_chorm = np.vstack([staturated_chorm[best_staturated_id, :], SelCh])
        staturated_value = ea.bs2ri(staturated_chorm, staturated_FieldD)
        staturated_ObjV = staturated_flow_loss(staturated_value)
        best_staturated_id = np.argmin(staturated_ObjV)
        if staturated_ObjV[best_staturated_id] < best_loss:
            best_staturate = staturated_value[best_staturated_id]
            best_loss = staturated_ObjV[best_staturated_id]
            net.staturated_flow = best_staturate.reshape((link_number, 3))
        print("generation ", gen, " best staturated flow loss", staturated_ObjV[best_staturated_id])

        FitnV = ea.ranking(capacity_ObjV, maxormins=maxormins)
        SelCh = capacity_chorm[ea.selecting(selectStyle, FitnV, NIND-1), :]
        SelCh = ea.recombin(recStyle, SelCh, pc)
        SelCh = ea.mutate(mutStyle, Encoding, SelCh, capacity_pm)

        capacity_chorm = np.vstack([capacity_chorm[best_capacity_id, :], SelCh])
        capacity_value = ea.bs2ri(capacity_chorm, capacity_FieldD)
        capacity_ObjV = capacity_loss(capacity_value)
        best_capacity_id = np.argmin(capacity_ObjV)
        if capacity_ObjV[best_capacity_id] < best_loss:
            best_capacity = capacity_value[best_capacity_id]
            best_loss = capacity_ObjV[best_capacity_id]
            net.capacity[:-1] = best_capacity
        print("generation ", gen, " best capacity loss", capacity_ObjV[best_capacity_id])

        print("generateion ", gen, " best loss ", best_loss)
    end_time = time.time()

    print("得到的最优函数值为 ", best_loss)
    np.save("cross_generatic_solution.npy", [best_split, best_staturate, best_capacity])
    print("用时 ", end_time-start_time, "s")

if __name__ == "__main__":

    evolution()
