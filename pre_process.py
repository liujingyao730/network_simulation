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

dest1 = pd.DataFrame()
dest2 = pd.DataFrame()
dest3 = pd.DataFrame()
dest4 = pd.DataFrame()
dest5 = pd.DataFrame()
dest6 = pd.DataFrame()

dest = [dest1, dest2, dest3, dest4, dest5, dest6]

def net_resolve(net_file, pickle_file):

    net_tree = etree.parse(net_file)
    root = net_tree.getroot()

    edge_list = {}
    lane_list = {}
    junction = {}
    tlc = {}
    junction_connection = {}

    for child in root:
        
        if child.tag == "edge":
            if "from" in child.attrib.keys():
                edge_id = child.attrib["id"]
                edge_list[edge_id] = []
                for lane in child:
                    length = float(lane.attrib["length"])
                    lane_id = lane.attrib["id"]
                    lane_list[lane_id] = length
                    edge_list[edge_id].append(lane_id)

        elif child.tag == "tlLogic":
            tlc_id = child.attrib["id"]
            offset = float(child.attrib["offset"])
            tlc[tlc_id] = {}
            tlc[tlc_id]["offset"] = offset
            tlc[tlc_id]["phases"] = []
            for phase in child:
                duration = float(phase.attrib["duration"])
                state = phase.attrib["state"]
                tlc[tlc_id]["phases"].append([duration, state])
            
        elif child.tag == "junction":

            if child.attrib["type"] == "traffic_light":
                junction_id = child.attrib["id"]
                junction[junction_id] = {}
                junction[junction_id]["intLanes"] = child.attrib["intLanes"].split(" ")
                junction[junction_id]["incLanes"] = child.attrib["incLanes"].split(" ")
            
        elif child.tag == "connection":

            if "via" in child.attrib.keys():
                connection_id = child.attrib["via"]
                connection_edge = "_".join(connection_id.split("_")[:-1])
                connection_from = child.attrib["from"]
                connection_to = child.attrib["to"]
                connection_dir = child.attrib["dir"]
                if "tl" in child.attrib.keys():
                    connection_tlc = child.attrib["tl"]
                else:
                    from_lane = connection_from + "_" + child.attrib["fromLane"]
                    connection_tlc = junction_connection[from_lane]["tlc"]
                junction_connection[connection_id] = {"from":connection_from, "to":connection_to, "tlc":connection_tlc, "dir":connection_dir}

    junction_cell = {}
    ordinary_cell = {}
    connection = {}
    signal_connection = {}
    tlc_time = []
    pass_char = ['y', 'Y', 'g', 'G']

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
        ordinary_cell[edge]["lane_number"] = len(edge_list[edge])
        for i in range(cell_num):
            cell_id = edge + '-' + str(i)
            ordinary_cell[edge]["cell_id"].append(cell_id)
            cell_start += cell_length
            if former_cell is not None:
                connection[former_cell] = cell_id
            former_cell = cell_id
    tlc_index = 0
    for tlc_id in tlc.keys():
        
        offset = tlc[tlc_id]["offset"]
        dura = [phase[0] for phase in tlc[tlc_id]["phases"]]
        phases = [phase[1] for phase in tlc[tlc_id]["phases"]]
        cycle = sum(dura)
        tlc_time.append([cycle, offset])
        start = 0
        for i in range(len(dura)):
            phase = phases[i]
            time = dura[i]
            end = (start + time) % cycle
            if end == 0:
                end = cycle
            for index in range(len(phase)):
                if phase[index] in pass_char:
                    inter_lane = junction[tlc_id]["intLanes"][index]
                    from_edge = junction_connection[inter_lane]["from"]
                    to_edge = junction_connection[inter_lane]["to"]
                    con_dir = junction_connection[inter_lane]["dir"]
                    via_edge = "_".join(inter_lane.split("_")[:-1])
                    to_cell = to_edge + "-0"
                    if from_edge not in ordinary_cell.keys():
                        from_edge = junction_connection[from_edge+"_0"]["from"]
                    from_cell = ordinary_cell[from_edge]["cell_id"][-1]
                    
                    if from_cell not in signal_connection.keys():
                        signal_connection[from_cell] = {}
                    
                    if to_cell not in signal_connection[from_cell].keys():
                        signal_connection[from_cell][to_cell] = [start, end, tlc_index, con_dir]
                    else:
                        if start == signal_connection[from_cell][to_cell][1]:
                            signal_connection[from_cell][to_cell][1] = end
                        if end == signal_connection[from_cell][to_cell][0]:
                            signal_connection[from_cell][to_cell][0] = start
            start = (start + time) % cycle

        tlc_index += 1
    
    for junction_lane in junction_connection:
        junction_cell[junction_lane] = junction_connection[junction_lane]["tlc"]

    net_information = {
        "ordinary_cell": ordinary_cell,
        "junction_cell": junction_cell,
        "connection": connection,
        "signal_connection": signal_connection,
        "tlc_time": tlc_time
    }

    with open(pickle_file, 'wb') as f:
        pickle.dump(net_information, f)

    return net_information

def init_dataframe(ordinary_cell, junction_cell):

    global dest1, dest2, dest3, dest4, dest5, dest6, dest

    cells = list(set(junction_cell.values()))
    for edge in ordinary_cell.keys():
        cells.extend(ordinary_cell[edge]["cell_id"])

    dest1 = pd.DataFrame(columns=cells)
    dest2 = pd.DataFrame(columns=cells)
    dest3 = pd.DataFrame(columns=cells)
    dest4 = pd.DataFrame(columns=cells)
    dest5 = pd.DataFrame(columns=cells)
    dest6 = pd.DataFrame(columns=cells)

    dest = [dest1, dest2, dest3, dest4, dest5, dest6]

def add_time(time):

    global dest1, dest2, dest3, dest4, dest5, dest6, dest

    dest1.loc[time] = 0
    dest2.loc[time] = 0
    dest3.loc[time] = 0
    dest4.loc[time] = 0
    dest5.loc[time] = 0
    dest6.loc[time] = 0

    dest = [dest1, dest2, dest3, dest4, dest5, dest6]

def save_file(prefix):

    global dest1, dest2, dest3, dest4, dest5, dest6, dest

    dest1_file = os.path.join(data_fold, prefix+"_dest1.csv")
    dest2_file = os.path.join(data_fold, prefix+"_dest2.csv")
    dest3_file = os.path.join(data_fold, prefix+"_dest3.csv")
    dest4_file = os.path.join(data_fold, prefix+"_dest4.csv")
    dest5_file = os.path.join(data_fold, prefix+"_dest5.csv")
    dest6_file = os.path.join(data_fold, prefix+"_dest6.csv")

    dest1.to_csv(dest1_file)
    dest2.to_csv(dest2_file)
    dest3.to_csv(dest3_file)
    dest4.to_csv(dest4_file)
    dest5.to_csv(dest5_file)
    dest6.to_csv(dest6_file)

    print("data file have saved with prefix ", prefix)

def load_file(prefix):

    global dest1, dest2, dest3, dest4, dest5, dest6, dest

    dest1_file = os.path.join(data_fold, prefix+"_dest1.csv")
    dest2_file = os.path.join(data_fold, prefix+"_dest2.csv")
    dest3_file = os.path.join(data_fold, prefix+"_dest3.csv")
    dest4_file = os.path.join(data_fold, prefix+"_dest4.csv")
    dest5_file = os.path.join(data_fold, prefix+"_dest5.csv")
    dest6_file = os.path.join(data_fold, prefix+"_dest6.csv")

    dest1 = pd.read_csv(dest1_file, index_col=0)
    dest2 = pd.read_csv(dest2_file, index_col=0)
    dest3 = pd.read_csv(dest3_file, index_col=0)
    dest4 = pd.read_csv(dest4_file, index_col=0)
    dest5 = pd.read_csv(dest5_file, index_col=0)
    dest6 = pd.read_csv(dest6_file, index_col=0)

    dest = [dest1, dest2, dest3, dest4, dest5, dest6]


def fcd_resolve(fcd_file, net_information, prefix="default"):

    global dest1, dest2, dest3, dest4, dest5, dest6, dest

    ordinary_cell = net_information["ordinary_cell"]
    junction_cell = net_information["junction_cell"]
    connection = net_information["connection"]
    signal_connection = net_information["signal_connection"]
    end_list = list(utils.end_edge.keys())

    root = etree.iterparse(fcd_file, events=["start"])

    init_dataframe(ordinary_cell, junction_cell)

    formerstep = {}
    nowstep = {}

    for events, elem in root:

        if elem.tag == "timestep":

            time = float(elem.attrib["time"])
            if time % 10 == 0:
                print(time)
            add_time(time)
        
        elif elem.tag == "vehicle":

            vehicle_id = elem.attrib["id"]
            lane = elem.attrib["lane"]
            pos = float(elem.attrib["pos"])
            route_index = int(vehicle_id.split("_")[0])

            if lane in junction_cell.keys():
                cell_id = junction_cell[lane]
            else:
                edge = "_".join(lane.split("_")[:-1])
                cell_length = ordinary_cell[edge]["cell_length"]
                start_pos = cell_length
                for i in range(len(ordinary_cell[edge]["cell_id"])):
                    if start_pos > pos:
                        cell_id = ordinary_cell[edge]["cell_id"][i]
                        break
                    start_pos += cell_length
            
            dest_edge = utils.routes[route_index][-1]
            destination = end_list.index(dest_edge)

            dest[destination].loc[time, cell_id] += 1
        
        elem.clear()

    save_file(prefix)

def calculate_layout(net_file, pickle_file):

    net_information = net_resolve(net_file, pickle_file)

    ordinary_cell = net_information["ordinary_cell"]
    cell_list = []
    for edge in ordinary_cell:
        cell_list.extend(ordinary_cell[edge]["cell_id"])
    cell_index = {cell_list[i]: i for i in range(len(cell_list))}

    tree = etree.parse(net_file)
    root = tree.getroot()

    layout = {}

    for child in root:

        if child.tag == "edge" and child.attrib["id"] in ordinary_cell.keys():

            edge = child.attrib["id"]
            lane_number = ordinary_cell[edge]["lane_number"]
            divide_number = len(ordinary_cell[edge]["cell_id"]) + 1
            x = np.array([0., 0.])
            y = np.array([0., 0.])

            for lane in child:

                shape = lane.attrib["shape"]
                points = shape.split(" ")
                assert len(points) == 2
                top = points[0].split(",")
                bottom = points[1].split(",")
                x += np.array([float(top[0]), float(top[1])])
                y += np.array([float(bottom[0]), float(bottom[1])])
            
            x /= lane_number
            y /= lane_number

            for i in range(len(ordinary_cell[edge]["cell_id"])):
                cell = ordinary_cell[edge]["cell_id"][i]
                index = cell_index[cell]
                layout[index] = (y - x) * (i + 1) / divide_number + x

    return layout

if __name__ == "__main__":
    net_file = "intersection.net.xml"
    fcd_file = "fcd.xml"
    data_fold = "data"
    pickle_file = "test.xml"

    layout = calculate_layout(net_file, pickle_file)
