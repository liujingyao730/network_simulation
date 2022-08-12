from numpy.core.defchararray import index
from numpy.core.records import array
from numpy.lib.shape_base import column_stack
import pandas as pd
import torch
import pickle
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

from com_model import replaceable_model,dyn_embedding
from struct_ablation import single_attention, single_attention_non_gate, baseline
from gnn_conv import gnn_conv
from gman4sim import Gman_sim
from network import data_on_network
import dir_manage as d
from utils import sparselist_to_tensor, from_sparse_get_index, from_sparse_get_reverse_index
from rgb_heatmap import rgb_map
from heat_animation import generate_video, get_picture_layout

def test_model(args, data_set):

    # with open(os.path.join(d.cell_data_path, args["net_file"]), 'rb') as f:
    #     net_information = pickle.load(f)
    
    show_detail = args.get("show_detail", False)

    # data_set = data_on_network(net_information, args["destination"][0], args["prefix"], args)
    # data_set.normalize_data()

    inputs, adj_list = data_set.get_batch()
    target = inputs[:, args["init_length"]+1:, :, :]

    '''
    net_type = pp.calculate_layout("four_test2.net.xml", "test.pkl")
    for i in range(len(adj_list)):
        print(i)
        data_set.show_adj(adj_list[i+args["init_length"]+1], network_type=net_type, with_label=True, colors=np.around(target[0, i, :, 17], decimals=1))
    '''
    # model = GCN_GRU(args)
    
    model_type = args.get("model_type", "st_node_encoder")
    if model_type == "replaceable_model":
        model = replaceable_model(args)
    elif model_type == "dyn_embedding":
        model = dyn_embedding(args)
    elif model_type == "single_attention":
        model = single_attention(args)
    elif model_type == "single_attention_non_gate":
        model = single_attention_non_gate(args)
    elif model_type == "baseline":
        model = baseline(args)
    elif model_type == "gnn_conv":
        model = gnn_conv(args)
    elif model_type == "GMAN":
        SE_file = args.get("SE_file", "four_large_SE.txt")
        SE_file = os.path.join(d.cell_data_path, SE_file)
        model = Gman_sim(SE_file, args, bn_decay=0.1)
    else:
        raise NotImplementedError
    model_file = os.path.join(d.log_path, args["model_prefix"], str(args["model"])+'.tar')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    inputs = torch.Tensor(inputs)
    target = torch.Tensor(target)
    if model_type == "GMAN":
        adj_list = data_set.index
    else:
        adj_list = torch.Tensor(sparselist_to_tensor(adj_list))

    if args.get("expand", False):
        node_size = args["node_size"]
        inputs_size = list(inputs.shape)
        inputs_size[2] = node_size
        time_size = adj_list.shape[0]
        inputs = torch.randn(inputs_size)
        target = torch.randn(inputs_size)
        adj_list = torch.randn([time_size, node_size, node_size])

    if args["gnn"] == "gat":
        index_list, weight_list = from_sparse_get_index(adj_list)
        reverse_index_list, reverse_weight_list = from_sparse_get_reverse_index(adj_list)
        weight_list = torch.Tensor(weight_list)
        reverse_weight_list = torch.Tensor(reverse_weight_list)
        weight_list = weight_list.cuda()
        reverse_weight_list = reverse_weight_list.cuda()

    if model_type != "GMAN":
        adj_list = adj_list.cuda()
    inputs = inputs.cuda()
    target = target.cuda()
    model = model.cuda()
    cell_index = data_set.name_to_id(args["input_cells_name"])
    model.set_input_cells(cell_index)
    time1 = time.time()
    if args["gnn"] == "gat":
        output = model.infer(inputs, adj_list, [index_list, reverse_index_list], [weight_list, reverse_weight_list])
    else:
        output = model.infer(inputs, adj_list)
    time2 = time.time()

    # target = data_set.recovery_data(target)
    # output = data_set.recovery_data(output)

    target = target.detach().cpu().numpy()
    output = output.detach().cpu().numpy()
    torch.cuda.empty_cache()

    print("计算用时 ", time2 - time1)
    if args.get("expand", False):
        return

    if show_detail:
        
        real_data = target[0, :, :, :]
        predict_data = output[0, :, :, :]
        start = args.get("show_start", 0)
        end = args.get("show_end", 400)
        show_cell = args.get("show_cell", None)

        for i in range(8):
            
            real_cell = real_data[start:end, show_cell, i]
            predict_cell = predict_data[start:end, show_cell, i]

            x = np.array(range(real_cell.shape[0]))

            plt.figure(figsize=(10,4))
            plt.plot(x, real_cell, label="gt")
            plt.plot(x, predict_cell, label="pd")
            plt.legend()
            plt.savefig("dest" + str(i) + ".png")

    show_3d = args.get("3D_show", False)
    if show_3d:

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        start_time = args["3D_start"]
        end_time = args["3D_end"]
        
        output_x = output[0, start_time:end_time, 33, args['3D_cell'][0]]
        output_y = output[0, start_time:end_time, 33, args['3D_cell'][1]]
        output_z = output[0, start_time:end_time, 33, args['3D_cell'][2]]

        target_x = target[0, start_time:end_time, 33, args['3D_cell'][0]]
        target_y = target[0, start_time:end_time, 33, args['3D_cell'][1]]
        target_z = target[0, start_time:end_time, 33, args['3D_cell'][2]]

        ax.plot(output_x, output_y, output_z, label="output")
        ax.plot(target_x, target_y, target_z, label="target")

        ax.legend()
        plt.savefig("3d_result.png")
    
    rgb_heatmap = args.get("rgb_heatmap", False)
    if rgb_heatmap:

        cells = args["cells"]
        start_t = args["rgb_start"]
        end_t = args["rgb_end"]
        dests = args["dests"]

        if len(dests) > 3:
            raise Exception('最多只能显示3个终点分量')

        inputs1 = output[0, start_t:end_t, cells, :]
        inputs1 = inputs1[:, :, dests]
        rgb_map(inputs1, output_file='output_rgb_heat.png')
        inputs2 = target[0, start_t:end_t, cells, :args["output_size"]]
        inputs2 = inputs2[:, :, dests]
        rgb_map(inputs2, output_file="target_rgb_heat.png")
        inputs3 = np.abs(inputs1-inputs2)
        rgb_map(inputs3, output_file="error_rate_file.png", is_error=True)
    
    heatmap = args.get("heat_map", False)
    if heatmap:

        cells = args["cells"][::-1]
        start = args["heatmap_start"]
        end = args["heatmap_end"]

        output_heat = np.sum(output[0, start:end, cells, :], axis=2)
        target_heat = np.sum(target[0, start:end, cells, :args["output_size"]], axis=2)
        col = range(start, end)
        output_heat = pd.DataFrame(output_heat, index=cells, columns=col)
        target_heat = pd.DataFrame(target_heat, index=cells, columns=col)

        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(output_heat, cmap='YlGnBu', linewidths=.5, ax=ax, xticklabels=10, vmax=40)
        plt.xlabel("time step", fontsize=20)
        plt.ylabel("nodes", fontsize=20)
        plt.savefig("output_heat.png", bbox_inches="tight")
        plt.cla()
        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(target_heat, cmap='YlGnBu', linewidths=.5, ax=ax, xticklabels=10, vmax=40)
        plt.xlabel("time step", fontsize=20)
        plt.ylabel("nodes", fontsize=20)
        plt.savefig("target_heat.png", bbox_inches="tight")
        plt.cla()

    output = np.sum(output, axis=3)
    target = np.sum(target[:, :, :, :args["output_size"]], axis=3)

    max_error = 0
    max_error_cell = -1

    for i in range(output.shape[2]):
        error = mean_squared_error(output[0, :, i], target[0, :, i])
        if error > 25:
            print(i, error)
        if error > max_error:
            max_error = error
            max_error_cell = i

    print("max error ", max_error, " in cell ", max_error_cell)
    ave_error = mean_squared_error(output[0, :, :], target[0, :, :])
    last_error = mean_squared_error(output[0, -1, :], target[0, -1, :])
    print(ave_error)
    print(last_error)
    # print(mean_absolute_error(output[0, :, :], target[0, :, :]) / np.mean(target[0, :, :]))

    get_eva_time = args.get("get_eva_time", False)
    if get_eva_time:
        for i in range(args["temporal_length"]-300, args["temporal_length"]):
            if np.max(output[0, i, :]) < 1:
                output_empty = i
                break
        for i in range(args["temporal_length"]-300, args["temporal_length"]):
            if np.max(target[0, i, :]) < 1:
                target_empty = i
                break

        print("output empty at ", output_empty*args["deltaT"])
        print("target empty at ", target_empty*args["deltaT"])

    if args.get("plot", False):
        show_cell = args.get("show_cell", None)
        start = args.get("show_start", 0)
        end = args.get("show_end", 400)
        if show_cell is None:
            real_cell = target[0, start:end, :].sum(1)
            predict_cell = output[0, start:end, :].sum(1)
        else:
            real_cell = target[0, start:end, show_cell]
            predict_cell = output[0, start:end, show_cell]
        x = np.array(range(real_cell.shape[0]))
        
        # np.save("four4_32.npy", real_cell)
        # np.save("four4_stgat_32.npy", predict_cell)

        plt.figure(figsize=(10,4))
        plt.plot(x, real_cell)
        plt.plot(x, predict_cell, label="our model")
        plt.legend()
        plt.xlabel("time step")
        plt.ylabel("vehicle number")
        plt.savefig("123.png")
    
    if args.get("save_output", False):
        save_data = output[0, :, :]
        save_data = save_data[:, args["save_cells"]].sum(1)
        test_target = target[0, :, :]
        test_target = test_target[:, args["save_cells"]].sum(1)
        np.save("output.npy", save_data)
        np.save("test_target.npy", test_target)
    
    if args.get("save_net_total", False):
        outputf = args["model_output_file"]
        targetf = args["target_output_file"]
        np.save(outputf, output[0, :, 32])
        np.save(targetf, target[0, :, 32])
    
    if args.get("is_heat_animation", False):
        net_xml_file = args.get("net_xml_file", "four_large.net.xml")
        edge_info, cell_info = get_picture_layout(net_xml_file)
        start = args.get("ani_start", 0)
        end = args.get("ani_end", 400)
        generate_video(target[0, start:end, :], output[0, start:end, :], edge_info, cell_info, output_title=args.get("model_name", "SimNET"))

    return ave_error


if __name__ == "__main__":

    basic_conf = os.path.join(d.config_data_path, "four_large_test.yaml")

    with open(basic_conf, 'rb') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(d.cell_data_path, args["net_file"]), 'rb') as f:
        net_information = pickle.load(f)
    data_set = data_on_network(net_information, args["destination"][0], args["prefix"], args)

    with torch.no_grad():
        test_model(args, data_set)
