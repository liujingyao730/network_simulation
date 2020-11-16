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

from model import GCN_GRU, node_encode_attention
from network import network_data
import dir_manage as d
from utils import sparselist_to_tensor
from loss_function import non_negative_loss, narrow_output_loss


def run_epoch(args, model, loss_function, optimizer, meter, sample_rate):

    prefix_list = args.get("prefix", ["default"])
    destination = args.get("destination", ["-genE0-2", "gneE3-2", "gneE4-2", "gneE5-2", "gneE2-2", "gneE6-2"])
    net_file = args.get("net_file", "test_net.pkl")
    input_cells_name = args.get("input_cells_name", ["gneE0-0", "-gneE3-0", "-gneE4-0", "-gneE5-0", "-gneE2-0", "-gneE6-0"])
    use_cuda = args.get("use_cuda", True)
    grad_clip = args.get("grad_clip", 10)
    show_every = args.get("show_every", 25)

    if isinstance(net_file, list):
        assert len(net_file) == len(prefix_list)
    elif isinstance(net_file, dict):
        net_file = [net_file[prefix] for prefix in prefix_list]
    elif isinstance(net_file, str):
        net_file = [net_file for prefix in prefix_list]
    
    if not isinstance(destination[0], list):
        destination = [destination for i in range(len(prefix_list))]

    if not isinstance(input_cells_name[0], list):
        input_cells_name = [input_cells_name for i in range(len(prefix_list))]

    batch_index = 0

    for i in range(len(prefix_list)):

        with open(os.path.join(d.cell_data_path, net_file[i]), 'rb') as f:
            net_information = pickle.load(f)

        data_set = network_data(net_information, destination[i], prefix_list[i], args)
        # data_set.normalize_data()
        input_cells = data_set.name_to_id(input_cells_name[i])
        model.set_input_cells(input_cells)

        while True:

            model.zero_grad()
            optimizer.zero_grad()

            data, adj_list = data_set.get_batch()

            adj_list = Variable(torch.Tensor(sparselist_to_tensor(adj_list)))
            inputs = Variable(torch.Tensor(data))
            targets = inputs[:, args["init_length"]:, :, :]

            if use_cuda:
                adj_list = adj_list.cuda()
                targets = targets.cuda()
                inputs = inputs.cuda()

            if random.random() > sample_rate:
                outputs = model.infer(inputs, adj_list)
            else:
                outputs = model(inputs, adj_list)

            loss = loss_function(torch.sum(targets[:, :, :, :args["output_size"]], dim=3), torch.sum(outputs, dim=3))

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

    prefix_list = args.get("test_prefix", ["default"])
    destination = args.get("test_destination", ["-genE0-2", "gneE3-2", "gneE4-2", "gneE5-2", "gneE2-2", "gneE6-2"])
    net_file = args.get("test_net_file", "test_net.pkl")
    input_cells_name = args.get("test_input_cells_name", ["gneE0-0", "-gneE3-0", "-gneE4-0", "-gneE5-0", "-gneE2-0", "-gneE6-0"])
    use_cuda = args.get("use_cuda", True)
    show_every = args.get("show_every", 25)

    if isinstance(net_file, list):
        assert len(net_file) == len(prefix_list)
    elif isinstance(net_file, dict):
        net_file = [net_file[prefix] for prefix in prefix_list]
    elif isinstance(net_file, str):
        net_file = [net_file for prefix in prefix_list]

    if not isinstance(destination[0], list):
        destination = [destination for i in range(len(prefix_list))]
    
    if not isinstance(input_cells_name[0], list):
        input_cells_name = [input_cells_name for i in range(len(prefix_list))]
    
    batch_index = 0

    for i in range(len(prefix_list)):

        with open(os.path.join(d.cell_data_path, net_file[i]), 'rb') as f:
            net_information = pickle.load(f)

        data_set = network_data(net_information, destination[i], prefix_list[i], args)
        # data_set.normalize_data()
        input_cells = data_set.name_to_id(input_cells_name[i])
        model.set_input_cells(input_cells)

        while True:
 
            inputs, adj_list = data_set.get_batch()

            adj_list = torch.Tensor(sparselist_to_tensor(adj_list))
            inputs = torch.Tensor(inputs)
            targets = inputs[:, args["init_length"]:, :, :]
 
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
    model = node_encode_attention(args)

    train_loss_function = narrow_output_loss(40)
    test_loss_function = torch.nn.MSELoss()

    if args["use_cuda"]:
        model = model.cuda()
        test_loss_function = test_loss_function.cuda()
        train_loss_function = train_loss_function.cuda()
    
    optimizer = torch.optim.Adagrad(model.parameters(), weight_decay=args["weight_decay"])

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

        meter = run_epoch(args, model, train_loss_function, optimizer, meter, sample_rate)

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

    parser.add_argument("--config", type=str, default="only_two")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    
    main()
