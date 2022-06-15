from email.policy import default
import numpy as np
from pandas import array
from scipy import interpolate
import matplotlib.pyplot as plt
import pickle
import xml.etree.cElementTree as etree
from matplotlib.collections import LineCollection
import cv2
import os

from pre_process import calculate_layout, net_resolve, enlarge_gap
from dir_manage import root_path

width = 10

def get_picture_layout(net_file):
    
    net_information = net_resolve(net_file, "default.pkl")

    ordinary_cell = net_information["ordinary_cell"]
    cell_list = []
    for edge in ordinary_cell:
        cell_list.extend(ordinary_cell[edge]["cell_id"])
    cell_index = {cell_list[i]: i for i in range(len(cell_list))}

    tree = etree.parse(net_file)
    root = tree.getroot()

    edge_info = {}
    cell_info = {}

    for child in root:

        if child.tag == "edge" and child.attrib["id"] in ordinary_cell.keys():

            edge = child.attrib["id"]
            lane_number = ordinary_cell[edge]["lane_number"]
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

            edge_info[edge] = [x, y]
        
    enlarge_gap(edge_info, 50, edge_info.keys())

    for edge in ordinary_cell.keys():
        
        edge_info[edge].append(len(ordinary_cell[edge]["cell_id"]))
        length = np.linalg.norm(edge_info[edge][0] - edge_info[edge][1], 2)
        edge_info[edge].append(length)
        divide_number = len(ordinary_cell[edge]["cell_id"]) * 2
        for i in range(len(ordinary_cell[edge]["cell_id"])):
            cell = ordinary_cell[edge]["cell_id"][i]
            index = cell_index[cell]
            cell_info[index] = [edge, i, int(length * (2 * i + 1) / divide_number)]
    
    return edge_info, cell_info


def single_frame(data, edge_info, cell_info, file="roadheat.png"):
    
    base_x = {}
    base_index = {}
    for edge in edge_info.keys():
        base_x[edge] = np.zeros(edge_info[edge][-2])
        base_index[edge] = []
    
    for cell in cell_info.keys():
        corr_edge = cell_info[cell][0]
        base_x[corr_edge][cell_info[cell][1]] = cell_info[cell][2]
        base_index[corr_edge].append(cell)
    
    fig, axs = plt.subplots(1, 1)
    
    for edge in edge_info.keys():
        x1, y1 = edge_info[edge][0]
        x2, y2 = edge_info[edge][1]
        x = np.linspace(x1, x2, 50)
        y = np.linspace(y1, y2, 50)
        
        if x1 > x2:
            a = 1
        if x1 != x2:
            color_x = np.abs(x-x1) * np.sqrt((y1-y2)**2 + (x1-x2)**2) / np.sqrt((x1-x2)**2)
        else:
            color_x = np.abs(y-y1)
        base_x_edge = np.concatenate((np.array([0]), base_x[edge], np.array([edge_info[edge][-1]])))
        base_y = data[base_index[edge]]
        base_y = np.concatenate((base_y[:1], base_y, base_y[-1:]))
        newf = interpolate.interp1d(base_x_edge, base_y, kind="cubic")
        color = newf(color_x)
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(color.min(), color.max())
        lc = LineCollection(segments, norm=norm)
        # Set the values used for colormapping
        lc.set_array(color)
        lc.set_linewidth(4)
        line = axs.add_collection(lc)
    
    axs.set_xlim(-1000, 1100)
    axs.set_ylim(-1000, 1000)
    # fig.colorbar(line, ax=axs)
    plt.savefig(file)
    plt.close()

def generate_video(data, edge_info, cell_info):
    
    tmp_picture_folder = os.path.join(root_path, "tmp_picture")
    if not os.path.exists(tmp_picture_folder):
        os.makedirs(tmp_picture_folder)
    
    frame, cell = data.shape
    
    for i in range(frame):
        picture_file = os.path.join(tmp_picture_folder, str(i)+'.png')
        single_frame(data[i, :], edge_info, cell_info, picture_file)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("heat_road_map.avi", fourcc, 24, (640, 480), True)
    
    for i in range(frame):
        frame = cv2.imread(os.path.join(tmp_picture_folder, str(i)+'.png'))
        video_writer.write(frame)
        os.remove(os.path.join(tmp_picture_folder, str(i)+'.png'))
    
    video_writer.release()
    cv2.destroyAllWindows()
   

if __name__ == "__main__":
    net_file = "four_large.net.xml"
    pickle_file = "test.pkl"

    with open("test.pkl", 'rb') as f:
        net_information = pickle.load(f)
    edge_info, cell_info = get_picture_layout(net_file)
    data = np.random.rand(48, 110) * 100
    # single_frame(data[0, :], edge_info, cell_info)
    generate_video(data, edge_info, cell_info)
