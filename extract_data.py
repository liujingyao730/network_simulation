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


net_file = "intersection.net.xml"
fcd_file = "fcd.xml"

def net_resovle(file):
    
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
                    inter_list[edge_id] = length
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
                        signal_connection[from_cell][to_cell] = [start, end]
            start = (start + time) % cycle

    return {"ordinary_cell":ordinary_cell, "inter_cell":inter_cell, "connection":connection, "signal_connection":signal_connection}



def fcd_resolve(fcd_file, cell_list):

    root = etree.iterparse(fcd_file, events=["start"])

    formerstep = {}
    nowstep = {}

    flow_in = pd.DataFrame(columns=["Nan"])
    flow_out = pd.DataFrame(columns=["Nan"])
    on_ramp = pd.DataFrame(columns=["Nan"])
    off_ramp = pd.DataFrame(columns=["Nan"])




net_information = net_resovle(net_file)