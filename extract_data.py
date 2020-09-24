import xml.etree.cElementTree as etree
import pandas as pd
import os 
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import xml.dom.minidom as dom
import yaml
import pickle

import utils

net_file = "intersection.net.xml"
fcd_file = "fcd.xml"
data_fold = "data"

flow_in = pd.DataFrame()
flow_out = pd.DataFrame()
left_in = pd.DataFrame()
left_out = pd.DataFrame()
right_in = pd.DataFrame()
right_out = pd.DataFrame()
dest1 = pd.DataFrame()
dest2 = pd.DataFrame()
dest3 = pd.DataFrame()
dest4 = pd.DataFrame()
dest5 = pd.DataFrame()
dest6 = pd.DataFrame()
direct = pd.DataFrame()
right = pd.DataFrame()
left = pd.DataFrame()
dest = [dest1, dest2, dest3, dest4, dest5, dest6]
di = [direct, right, left]
In = [flow_in, right_in, left_in]
Out = [flow_out, right_out, left_out]

def net_resovle(file, pickle_file):
    
    net_tree = etree.parse(net_file)
    root = net_tree.getroot()

    edge_list = {}
    lane_list = {}
    inter_list = {}
    junction_connection = {}
    junction = {}
    signal = {}

    for child in root:
        if child.tag == "edge":
            if "function" in child.attrib.keys() and child.attrib["function"] == "internal":
                edge_id = child.attrib["id"]
                for lane in child:
                    lane_id = lane.attrib["id"]
                    length = float(lane.attrib["length"])
                    inter_list[edge_id] = {"length":length}
            else:
                edge_id = child.attrib["id"]
                edge_list[edge_id] = []
                for lane in child:
                    length = lane.attrib["length"]
                    lane_id = lane.attrib["id"]
                    lane_list[lane_id] = float(length)
                    edge_list[edge_id].append(lane_id)
        elif child.tag == "tlLogic":
            tlc_id = child.attrib["id"]
            offset = float(child.attrib["offset"])
            signal[tlc_id] = {}
            signal[tlc_id]["offset"] = offset
            signal[tlc_id]["phases"] = []
            for phase in child:
                dura = float(phase.attrib["duration"])
                state = phase.attrib["state"]
                signal[tlc_id]["phases"].append([dura, state])
        elif child.tag == "junction":
            if child.attrib["type"] == "traffic_light":
                junction_id = child.attrib["id"]
                junction[junction_id] = child.attrib["intLanes"].split(" ")
        elif child.tag == "connection":
            if "tl" in child.attrib.keys():
                connection_tlc = child.attrib["tl"]
                connection_id = child.attrib["via"]
                connection_from = child.attrib["from"]
                connection_to = child.attrib["to"]
                connection_type = child.attrib["dir"]
                junction_connection[connection_id] = {"from":connection_from, "to":connection_to, "type":connection_type, "tlc":connection_tlc}

    inter_cell = inter_list
    ordinary_cell = {}
    connection = {}
    signal_connection = {}
    pass_char = ['y', 'Y', 'G', 'g']
    for edge in edge_list.keys():
        ordinary_cell[edge] = {}
        lane = edge_list[edge][0]
        length = lane_list[lane]
        cell_num = round(length / 100)
        cell_length = length / cell_num
        cell_start = 0
        former_cell = None
        ordinary_cell[edge]["cell_length"] = cell_length
        ordinary_cell[edge]["cell_id"] = []
        for i in range(cell_num):
            cell_id = edge + "-" + str(i)
            ordinary_cell[edge]["cell_id"].append(cell_id)
            cell_start += cell_length
            if former_cell is not None:
                connection[former_cell] = cell_id
            former_cell = cell_id

    for tlc in signal.keys():
        offset = signal[tlc]["offset"]
        dura = [phase[0] for phase in signal[tlc]["phases"]]
        phases = [phase[1] for phase in signal[tlc]["phases"]]
        cycle = sum(dura)
        signal_connection["cycle"] = cycle
        start = offset % cycle
        for i in range(len(dura)):
            phase = phases[i]
            time = dura[i]
            end = (start + time) % cycle
            if end == 0:
                end = cycle
            for index in range(len(phase)):
                if phase[index] in pass_char:
                    inter_lane = junction[tlc][index]
                    from_edge = junction_connection[inter_lane]["from"]
                    to_edge = junction_connection[inter_lane]["to"]
                    direction = junction_connection[inter_lane]["type"]
                    via_edge = "_"
                    via_edge = via_edge.join(inter_lane.split("_")[:-1])
                    inter_cell[via_edge]["from_edge"] = from_edge
                    to_cell = to_edge + '-0'
                    from_cell = ordinary_cell[from_edge]["cell_id"][-1]
                    if from_cell not in signal_connection.keys():
                        signal_connection[from_cell] = {}

                    if to_cell in signal_connection[from_cell].keys():
                        if start == signal_connection[from_cell][to_cell][0] and end == signal_connection[from_cell][to_cell][1]:
                            continue

                    if phase[index] in ["y", "Y"]:
                        signal_connection[from_cell][to_cell][1] = end
                    else:   
                        signal_connection[from_cell][to_cell] = [start, end, direction]
            start = (start + time) % cycle

    with open(pickle_file, 'wb') as f:
        pickle.dump(net_information, pickle_file)

    return {"ordinary_cell":ordinary_cell, "inter_cell":inter_cell, "connection":connection, "signal_connection":signal_connection}


def init_dataframe(ordinary_cell, inter_cell):

    global flow_in, flow_out, left_in, left_out, right_in, right_out
    global dest1, dest2, dest3, dest4, dest5, dest6, right, left, direct, dest, di

    cells = list(inter_cell.keys())
    for edge in ordinary_cell.keys():
        cells.extend(ordinary_cell[edge]["cell_id"])

    flow_in = pd.DataFrame(columns=cells)
    flow_out = pd.DataFrame(columns=cells)
    left_in = pd.DataFrame(columns=cells)
    left_out = pd.DataFrame(columns=cells)
    right_in = pd.DataFrame(columns=cells)
    right_out = pd.DataFrame(columns=cells)
    dest1 = pd.DataFrame(columns=cells)
    dest2 = pd.DataFrame(columns=cells)
    dest3 = pd.DataFrame(columns=cells)
    dest4 = pd.DataFrame(columns=cells)
    dest5 = pd.DataFrame(columns=cells)
    dest6 = pd.DataFrame(columns=cells)
    direct = pd.DataFrame(columns=cells)
    right = pd.DataFrame(columns=cells)
    left = pd.DataFrame(columns=cells)

    dest = [dest1, dest2, dest3, dest4, dest5, dest6]
    di = [direct, right, left]

def add_time(time):

    global flow_in, flow_out, left_in, left_out, right_in, right_out
    global dest1, dest2, dest3, dest4, dest5, dest6, right, left, direct, dest, di

    flow_in.loc[time] = 0
    flow_out.loc[time] = 0
    left_out.loc[time] = 0
    right_out.loc[time] = 0
    left_in.loc[time] = 0
    right_in.loc[time] = 0
    dest1.loc[time] = 0
    dest2.loc[time] = 0
    dest3.loc[time] = 0
    dest4.loc[time] = 0
    dest5.loc[time] = 0
    dest6.loc[time] = 0
    direct.loc[time] = 0
    right.loc[time] = 0
    left.loc[time] = 0

    dest = [dest1, dest2, dest3, dest4, dest5, dest6]
    di = [direct, right, left]

def save_file(prefix):

    global flow_in, flow_out, left_in, left_out, right_in, right_out
    global dest1, dest2, dest3, dest4, dest5, dest6, right, left, direct, dest, di

    flow_in_file = os.path.join(data_fold, prefix + "_flow_in.csv")
    flow_out_file = os.path.join(data_fold, prefix + "_flow_out.csv")
    left_in_file = os.path.join(data_fold, prefix + "_left_in.csv")
    left_out_file = os.path.join(data_fold, prefix + "_left_out.csv")
    right_in_file = os.path.join(data_fold, prefix + "_right_in.csv")
    right_out_file = os.path.join(data_fold, prefix + "_right_out.csv")
    dest1_file = os.path.join(data_fold, prefix+"_dest1.csv")
    dest2_file = os.path.join(data_fold, prefix+"_dest2.csv")
    dest3_file = os.path.join(data_fold, prefix+"_dest3.csv")
    dest4_file = os.path.join(data_fold, prefix+"_dest4.csv")
    dest5_file = os.path.join(data_fold, prefix+"_dest5.csv")
    dest6_file = os.path.join(data_fold, prefix+"_dest6.csv")
    direct_file = os.path.join(data_fold, prefix+"_direct.csv")
    right_file = os.path.join(data_fold, prefix+"_right.csv")
    left_file = os.path.join(data_fold, prefix+"_left.csv")

    flow_in.to_csv(flow_in_file)
    flow_out.to_csv(flow_out_file)
    left_out.to_csv(left_out_file)
    right_out.to_csv(right_out_file)
    left_in.to_csv(left_in_file)
    right_in.to_csv(right_in_file)
    dest1.to_csv(dest1_file)
    dest2.to_csv(dest2_file)
    dest3.to_csv(dest3_file)
    dest4.to_csv(dest4_file)
    dest5.to_csv(dest5_file)
    dest6.to_csv(dest6_file)
    direct.to_csv(direct_file)
    right.to_csv(right_file)
    left.to_csv(left_file)

    print("data file have saved with prefix ", prefix)

def fcd_resolve(fcd_file, net_information, prefix="defualt"):

    global flow_in, flow_out, left_in, left_out, right_in, right_out
    global dest1, dest2, dest3, dest4, dest5, dest6, right, left, direct, dest, di

    ordinary_cell = net_information["ordinary_cell"]
    inter_cell = net_information["inter_cell"]
    connection = net_information["connection"]
    signal_connection = net_information["connection"]
    
    end_list = list(utils.end_edge.keys())
    direction_list = ['d', 'r', 'l']

    root = etree.iterparse(fcd_file, events=["start"])

    init_dataframe(ordinary_cell, inter_cell)

    formerstep = {}
    nowstep = {}

    time = -1

    for events, elem in root:

        if elem.tag == "timestep":
            
            for vehicle in nowstep.keys():

                if vehicle not in formerstep.keys():
                    flow_in.loc[time, nowstep[vehicle]] += 1
                else:
                    n_cell = nowstep[vehicle]
                    f_cell = formerstep[vehicle]
                    if f_cell != n_cell:
                        if f_cell in signal_connection.keys() and n_cell in signal_connection[f_cell].keys():
                            if signal_connection[f_cell][n_cell][2] == 'l':
                                left_in.loc[time, n_cell] += 1
                                left_out.loc[time, f_cell] += 1
                            elif signal_connection[f_cell][n_cell] == 'r':
                                right_in.loc[time, n_cell] += 1
                                left_out.loc[time, f_cell] += 1
                            elif signal_connection[f_cell][n_cell] == 'd':
                                flow_in.loc[time, n_cell] += 1
                                flow_out.loc[time, f_cell] += 1
                            else:
                                raise Exception("Unkown connection type ", signal_connection[f_cell][n_cell])
                        else:
                            flow_in.loc[time, n_cell] += 1
                            flow_out.loc[time, f_cell] += 1
                
                    formerstep.pop(vehicle)
                
            for vehicle in formerstep.keys():
                flow_out.loc[time, formerstep[vehicle]] += 1
                
            formerstep = nowstep.copy()
            nowstep.clear()

            time = float(elem.attrib["time"])
            add_time(time)
        elif elem.tag == "vehicle":
            vehicle_id = elem.attrib["id"]
            lane = elem.attrib["lane"]
            pos = float(elem.attrib["pos"])
            route_index = int(vehicle_id.split("_")[0])

            edge = "_"
            edge = edge.join(lane.split("_")[:-1])
            if edge in ordinary_cell.keys():
                cell_length = ordinary_cell[edge]["cell_length"]
                start_pos = cell_length
                for i in range(len(ordinary_cell[edge]["cell_id"])):
                    if start_pos > pos:
                        cell_id = ordinary_cell[edge]["cell_id"][i]
                        break
                    start_pos += cell_length
            elif edge in inter_cell.keys():
                cell_id = edge
                edge = inter_cell[edge]["from_edge"]

            dest_edge = utils.routes[route_index][-1]
            destination = end_list.index(dest_edge)
            if dest_edge == edge:
                direction = 0
            else:
                edge_index = utils.routes[route_index].index(edge)
                next_edge = utils.routes[route_index][edge_index + 1]
                direction = direction_list.index(utils.edge_dir[edge][next_edge])

            dest[destination].loc[time, cell_id] += 1
            di[direction].loc[time, cell_id] += 1
            
        elem.clear()

    
    save_file(prefix)
    
                
def reset_data(prefix, )

net_information = net_resovle(net_file)
fcd_resolve(fcd_file, net_information)