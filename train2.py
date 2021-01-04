import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import numpy as np
import os
import yaml
import torchnet
import random
import time
import argparse

from model import GCN_GRU, node_encode_attention, node_encode_attention_res
from coder_model import st_node_encoder, st_node_encoder_res
from network import data_on_network
import dir_manage as d
from utils import sparselist_to_tensor
from loss_function import non_negative_loss, narrow_output_loss

def train_epoch(args, model, loss_function, optimizer, meter, sample_rate):

    prefix_list = args.get("prefix", [["default"]])
    destination = args.get("destination", [["-genE0-2", "gneE3-2", "gneE4-2", "gneE5-2", "gneE2-2", "gneE6-2"]])
    net_file = args.get("net_file", ["test_net.pkl"])
    input_cells_name = args.get("input_cells_name", [["gneE0-0", "-gneE3-0", "-gneE4-0", "-gneE5-0", "-gneE2-0", "-gneE6-0"]])
    use_cuda = args.get("use_cuda", True)
    grad_clip = args.get("grad_clip", 10)
    show_every = args.get("show_every", 100)

    batch_index = 0

    for i in range(len(net_file)):

        with open(os.path.join(d.cell_data_path, net_file[i]), 'rb') as f:
            net_information = pickle.load(f)

        data_set = data_on_network(net_information, destination[i], prefix_list[i], args)
        input_cells = data_set.name_to_id(input_cells_name[i])
        model.set_input_cells(input_cells)

        print("process net ", net_file[i])

        while True:

            model.zero_grad()
            optimizer.zero_grad()

            data, adj_list = data_set.get_batch()

            adj_list = Variable(torch.Tensor(sparselist_to_tensor(adj_list)))
            inputs = Variable(torch.Tensor(data))
            targets = inputs[:, args["init_length"]+1:, :, :]

            if use_cuda:
                adj_list = adj_list.cuda()
                targets = targets.cuda()
                inputs = inputs.cuda()

            if random.random() > sample_rate:
                outputs = model.infer(inputs, adj_list)
            else:
                outputs = model(inputs, adj_list)

            dest_loss = loss_function(targets[:, :, :, :args["output_size"]], outputs)

            total_loss = loss_function(
                torch.sum(targets[:, :, :, :args["output_size"]], dim=3), 
                torch.sum(outputs, dim=3)
            )

            loss = dest_loss + total_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            meter.add(loss.item())

            if batch_index % show_every == 0:

                print("batch {}, train_loss = {:.3f}".format(batch_index, meter.value()[0]))
            
            batch_index += 1

            if not data_set.next_index():
                break
    
    return meter

def test_epoch(args, model, loss_function, meter):

    prefix_list = args.get("test_prefix", [["default"]])
    destination = args.get("test_destination", [["-genE0-2", "gneE3-2", "gneE4-2", "gneE5-2", "gneE2-2", "gneE6-2"]])
    net_file = args.get("test_net_file", ["test_net.pkl"])
    input_cells_name = args.get("test_input_cells_name", [["gneE0-0", "-gneE3-0", "-gneE4-0", "-gneE5-0", "-gneE2-0", "-gneE6-0"]])
    use_cuda = args.get("use_cuda", True)
    show_every = args.get("show_every", 100)

    batch_index = 0

    for i in range(len(net_file)):

        with open(os.path.join(d.cell_data_path, net_file[i]), "rb") as f:
            net_information = pickle.load(f)
        
        data_set = data_on_network(net_information, destination[i], prefix_list[i], args)
        input_cells = data_set.name_to_id(input_cells_name[i])
        model.set_input_cells(input_cells)

        while True:
 
            inputs, adj_list = data_set.get_batch()

            adj_list = torch.Tensor(sparselist_to_tensor(adj_list))
            inputs = torch.Tensor(inputs)
            targets = inputs[:, args["init_length"]+1:, :, :]
 
            if use_cuda:
                adj_list = adj_list.cuda()
                targets = targets.cuda()
                inputs = inputs.cuda()

            outputs = model.infer(inputs, adj_list)
            
            # targets = data_set.recovery_data(targets)
            # outputs = data_set.recovery_data(outputs)

            loss = loss_function(torch.sum(targets[:, :, :, :args["output_size"]], dim=3), torch.sum(outputs, dim=3))

            meter.add(loss.item())

            if batch_index % show_every == 0:

                print("batch {}, test_loss = {:.3f}".format(batch_index, meter.value()[0]))
            
            batch_index += 1

            if not data_set.next_index():
                break
    
    return meter

def train(args):

    torch.autograd.set_detect_anomaly(True)

    with open(os.path.join(d.config_data_path, args.config+".yaml"), 'rb') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    record_fold = os.path.join(d.log_path, args.get("name", "default"))

    if not os.path.exists(record_fold):
        os.makedirs(record_fold)

    log_file = open(os.path.join(record_fold, "log.txt"), 'w+')

    for key in args.keys():
        print(key, " ", args[key])
        log_file.write(key+"  "+str(args[key])+'\n')

    # model = GCN_GRU(args)
    # model = node_encode_attention_res(args)
    model = st_node_encoder(args)

    train_loss_function = narrow_output_loss(60)
    # train_loss_function = non_negative_loss()
    test_loss_function = torch.nn.MSELoss()

    if args["use_cuda"]:
        model = model.cuda()
        test_loss_function = test_loss_function.cuda()
        train_loss_function = train_loss_function.cuda()
    
    optimizer = torch.optim.Adam(model.parameters())

    meter = torchnet.meter.AverageValueMeter()

    best_epoch = -1
    best_loss = float('inf')
    sample_rate = args.get("sample_rate", 1)
    sample_decay = args.get("sample_decay", 0.04)

    for epoch in range(args.get("num_epochs", 50)):

        print("training epoch begins")

        model.train()
        meter.reset()

        start = time.time()

        meter = train_epoch(args, model, train_loss_function, optimizer, meter, sample_rate)

        end1 = time.time()

        sample_rate -= sample_decay

        print("epoch{}, training loss = {:.3f}, time consuming {:.2f}".format(epoch, meter.value()[0], end1 - start))
        log_file.write("epoch{}, training loss = {:.3f}, time consuming {:.2f}\n".format(epoch, meter.value()[0], end1 - start))

        meter.reset()
        model.eval()

        meter = test_epoch(args, model, test_loss_function, meter)

        end2 = time.time()

        print("epoch{}, test loss = {:.3f}, time consuming {:.2f}".format(epoch, meter.value()[0], end2 - end1))
        log_file.write("epoch{}, test loss = {:.3f}, time consuming {:.2f}\n".format(epoch, meter.value()[0], end2 - end1))

        if meter.value()[0] < best_loss:
            best_epoch = epoch
            best_loss = meter.value()[0]

        torch.save({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, os.path.join(record_fold, str(epoch)+'.tar'))
    
    print("best epoch {}, best loss {}".format(best_epoch, best_loss))
    log_file.write("best epoch {}, best loss {}\n".format(best_epoch, best_loss))

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="four_var_train2")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    
    main()