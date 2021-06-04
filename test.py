import torch
import pickle
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt

from model import GCN_GRU, node_encode_attention
from coder_model import st_node_encoder, coder_on_dir
from com_model import replaceable_model,dyn_embedding
from struct_ablation import single_attention, single_attention_non_gate, baseline
from gnn_conv import gnn_conv
from feature_ablation import non_dir_model
from network import network_data, data_on_network
import dir_manage as d
from utils import sparselist_to_tensor, from_sparse_get_index, from_sparse_get_reverse_index
import pre_process as pp

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
    else:
        raise NotImplementedError
    model_file = os.path.join(d.log_path, args["model_prefix"], str(args["model"])+'.tar')
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    inputs = torch.Tensor(inputs)
    target = torch.Tensor(target)
    adj_list = torch.Tensor(sparselist_to_tensor(adj_list))

    if args["gnn"] == "gat":
        index_list, weight_list = from_sparse_get_index(adj_list)
        reverse_index_list, reverse_weight_list = from_sparse_get_reverse_index(adj_list)
        weight_list = torch.Tensor(weight_list)
        reverse_weight_list = torch.Tensor(reverse_weight_list)
        weight_list = weight_list.cuda()
        reverse_weight_list = reverse_weight_list.cuda()

    inputs = inputs.cuda()
    target = target.cuda()
    adj_list = adj_list.cuda()
    model = model.cuda()
    cell_index = data_set.name_to_id(args["input_cells_name"])
    model.set_input_cells(cell_index)

    if args["gnn"] == "gat":
        output = model.infer(inputs, adj_list, [index_list, reverse_index_list], [weight_list, reverse_weight_list])
    else:
        output = model.infer(inputs, adj_list)

    # target = data_set.recovery_data(target)
    # output = data_set.recovery_data(output)

    if show_detail:
        
        real_data = target[0, :, :, :].detach().cpu().numpy()
        predict_data = output[0, :, :, :].detach().cpu().numpy()
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

    if args["3D_show"]:

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        start_time = args["3D_start"]
        end_time = args["3D_end"]
        
        output_x = output[0, start_time:end_time, 33, args['3D_cell'][0]].detach().cpu().numpy()
        output_y = output[0, start_time:end_time, 33, args['3D_cell'][1]].detach().cpu().numpy()
        output_z = output[0, start_time:end_time, 33, args['3D_cell'][2]].detach().cpu().numpy()

        target_x = target[0, start_time:end_time, 33, args['3D_cell'][0]].detach().cpu().numpy()
        target_y = target[0, start_time:end_time, 33, args['3D_cell'][1]].detach().cpu().numpy()
        target_z = target[0, start_time:end_time, 33, args['3D_cell'][2]].detach().cpu().numpy()

        ax.plot(output_x, output_y, output_z, label="output")
        ax.plot(target_x, target_y, target_z, label="target")

        ax.legend()
        plt.savefig("3d_result.png")

    f = torch.nn.MSELoss()
    output = torch.sum(output, dim=3)
    target = torch.sum(target[:, :, :, :args["output_size"]], dim=3)

    max_error = 0
    max_error_cell = -1

    for i in range(output.shape[2]):
        error = f(output[0, :, i], target[0, :, i])
        if error > 25:
            print(i, error)
        if error > max_error:
            max_error = error
            max_error_cell = i

    print("max error ", max_error, " in cell ", max_error_cell)
    ave_error = f(output, target)
    last_error = f(output[:, -1, :], target[:, -1, :])
    print(ave_error)
    print(last_error)

    get_eva_time = args.get("get_eva_time", False)
    if get_eva_time:
        for i in range(args["temporal_length"]-300, args["temporal_length"]):
            if torch.max(output[0, i, :]) < 1:
                output_empty = i
                break
        for i in range(args["temporal_length"]-300, args["temporal_length"]):
            if torch.max(target[0, i, :]) < 1:
                target_empty = i
                break

        print("output empty at ", output_empty*args["deltaT"])
        print("target empty at ", target_empty*args["deltaT"])

    if args.get("plot", False):
        show_cell = args.get("show_cell", None)
        start = args.get("show_start", 0)
        end = args.get("show_end", 400)
        if show_cell is None:
            real_cell = target[0, start:end, :].detach().cpu().numpy().sum(1)
            predict_cell = output[0, start:end, :].detach().cpu().numpy().sum(1)
        else:
            real_cell = target[0, :, show_cell].detach().cpu().numpy()[start:end]
            predict_cell = output[0, :, show_cell].detach().cpu().numpy()[start:end]
        x = np.array(range(real_cell.shape[0]))

        plt.figure(figsize=(10,4))
        plt.plot(x, real_cell, label="gt")
        plt.plot(x, predict_cell, label="pd")
        plt.legend()
        plt.savefig("123.png")

    return ave_error


if __name__ == "__main__":

    basic_conf = os.path.join(d.config_data_path, "four_large_test.yaml")

    with open(basic_conf, 'rb') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    with open(os.path.join(d.cell_data_path, args["net_file"]), 'rb') as f:
        net_information = pickle.load(f)
    data_set = data_on_network(net_information, args["destination"][0], args["prefix"], args)
    test_model(args, data_set)
