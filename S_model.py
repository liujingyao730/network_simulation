import numpy as np
import pickle
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

import dir_manage

class network(object):

    def __init__(self, net_pickle):

        with open(os.path.join(dir_manage.cell_data_path, net_pickle), "rb") as f:
            net_information = pickle.load(f)
        
        self.link_number = len(net_information["ordinary_cell"].keys())
        self.index2id = {i:list(net_information["ordinary_cell"].keys())[i] for i in range(self.link_number)}
        self.id2index = {list(net_information["ordinary_cell"].keys())[i]:i for i in range(self.link_number)}
        self.index2id[self.link_number] = "end_link"
        self.id2index["end_link"] = self.link_number

        self.vehicle_number = np.zeros(self.link_number)
        self.queue_length = np.zeros(self.link_number, 3)
        self.capacity = np.zeros(self.link_number)
        self.lane_number = np.zeros(self.link_number)
        self.free_speed = np.ones(self.link_number) * 16.67
        self.vehicle_length = 4.5

        time_storage = 3

        self.split_rate = np.zeros((self.link_number, 3))
        self.leave_flow = np.zeros((self.link_number, 3))

        self.enter_flow = np.zeros((self.link_number, time_storage))
        self.gamma = np.zeros((self.link_number, 2))
        self.sigma = np.zeros((self.link_number, 2))

        self.green_time = np.zeros((self.link_number, 3))
        self.cycle_time = np.zeros(self.link_number)

        self.split_rate = np.zeros((self.link_number, 3))
        self.arrive_flow = np.zeros((self.link_number, 3))
        self.leave_flow = np.zeros((self.link_number, 3))
        self.staturated_flow = np.zeros((self.link_number, 3))

        self.link_to = np.ones((self.link_number, 3, 2)) * -1
        self.link_from = np.ones((self.link_number, 3, 2)) * -1

        for i in range(self.link_number):
            self.lane_number[i] = net_information["ordinary_cell"][self.index2id[i]]["lane_number"]
            cell_length = net_information["ordinary_cell"][self.index2id[i]]["cell_length"]
            cell_number = len(net_information["ordinary_cell"][self.index2id[i]]["cell_id"])
            self.capacity[i] = self.lane_number[i] * cell_length * cell_number
        
        for from_cell in net_information["signal_connection"].keys():
            
            from_link = from_cell[:-2]
            from_link = self.id2index[from_link]
            i = 0
            
            for to_cell in net_information["signal_connection"][from_cell].keys():
                
                to_link = to_cell[:-2]
                to_link = self.id2index[to_link]
                
                tlc_index = net_information["signal_connection"][from_cell][to_cell][2]
                self.cycle_time[from_link] = net_information["tlc_time"][tlc_index][0]
                green_t = net_information["signal_connection"][from_cell][to_cell][1] - net_information["signal_connection"][from_cell][to_cell][0]
                if green_t <= 0:
                    green_t += self.cycle_time[from_link]
                
                for j in range(3):
                    if self.link_from[to_link, j, 0] == -1:
                        break
                
                self.link_to[from_link, i, 0] = to_link
                self.link_to[from_link, i, 1] = j
                self.link_from[to_link, j, 0] = from_link
                self.link_from[to_link, j, 1] = i
                self.green_time[from_link, i] = green_t

                i += 1
        
    def update_sigma_gamma(self, link):

        self.sigma[link, 0] = self.sigma[link, 1]
        self.gamma[link, 0] = self.sigma[link, 1]
        factor = (self.capacity[link] - np.sum(self.queue_length[link, :])) * self.vehicle_length / (self.lane_number[link] * self.free_speed[link])
        self.sigma[link, 1] = int(factor / self.cycle_time[link])
        self.gamma[link, 1] = factor - self.sigma[link, 1] * self.cycle_time[link]

    def update_enter(self, link):

        enter_flow = 0
        for i in range(3):
            f_link = self.link_from[link, i, 0]
            index = self.link_from[link, i, 1]
            if index == -1:
                break
            enter_flow += self.leave_flow[f_link, index]
        
        self.enter_flow[link, :-1] = self.enter_flow[link, 1:]
        self.enter_flow[link, -1] = enter_flow
    
    def update_arrive(self, link):

        self.update_sigma_gamma(link)
        sigma_k = int(self.sigma[link, 1])
        sigma_k_1 = int(self.sigma[link, 0])
        gamma_k = self.gamma[link, 1]
        gamma_k_1 = self.gamma[link, 0]
        enter_1 = self.enter_flow[link, sigma_k]
        enter_2 = self.enter_flow[link, sigma_k_1+1]
        arrive_flow_total = (self.cycle_time[link] - gamma_k) / self.cycle_time * enter_1
        arrive_flow_total += gamma_k_1 / self.cycle_time * enter_2

        for i in range(3):
            if self.link_from[link, i, 0] == -1:
                break
            self.arrive_flow[link, i] = arrive_flow_total * self.split_rate[link, i]

    def update_leave_to(self, link1, link2):

        for i in range(3):
            if self.link_to[link1, i, 0] == link2:
                break
        factor_1 = self.staturated_flow(link1) * self.green_time[link1, i] / self.cycle_time[link1]
        factor_2 = self.queue_length[link1, i] / self.cycle_time + np.sum(self.arrive_flow[link1, :])
        factor_3 = self.split_rate[link1, i](self.capacity[link1] - self.vehicle_number[link1])
        factor_3 = factor_3 / np.sum(self.split_rate[link1, :] * self.cycle_time[link1])

        self.arrive_flow[link1, i] = min(factor_1, factor_2, factor_3)
    
    def update_vehicle_number(self, link):

        self.vehicle_number[link] = self.vehicle_number[link] + (np.sum(self.enter_flow[link, :]) - np.sum(self.leave_flow[link, :])) * self.cycle_time[link]
    
    def update_queue_length(self, link):

        self.queue_length[link, 0] = self.queue_length[link, 0] + (self.arrive_flow[link, 0] - self.leave_flow[link, 0]) * self.cycle_time[link]
        self.queue_length[link, 1] = self.queue_length[link, 1] + (self.arrive_flow[link, 1] - self.leave_flow[link, 1]) * self.cycle_time[link]
        self.queue_length[link, 2] = self.queue_length[link, 2] + (self.arrive_flow[link, 2] - self.leave_flow[link, 2]) * self.cycle_time[link]

    def set_input_output(self, input_links, output_links):

        self.input_links = input_links
        for link in output_links:
            self.link_to[link, 0, 0] = self.link_number
            self.link_to[link, 0, 1] = 0
            self.link_to[link, 1, 0] = self.link_number
            self.link_to[link, 1, 1] = 0
            self.link_to[link, 2, 0] = self.link_number
            self.link_to[link, 2, 1] = 0
        
        self.normal_links = [i for i in range(self.link_number) if i not in self.input_links]
        
    def set_split_rate(self, link, rate):

        self.split_rate[link, :] = rate / np.sum(rate)

    def step(self, inputs):

        for link in self.input_links:
            link_id = self.index2id[link]
            self.enter_flow[link, 0] = inputs[link_id]

        for link in self.normal_links:
            self.update_enter(link)
        
        for link in range(self.link_number):
            self.update_arrive(link)
        
        for link in range(self.link_number):
            for i in range(3):
                if self.link_to[link, i, 0] == -1:
                    break
                to_link = self.link_to[link, i, 0]
                self.update_leave_to(link, to_link)
        
        for link in range(self.link_number):
            self.update_vehicle_number(link)
            self.update_queue_length(link)
    
    def calculate_loss(self, inputs, targets):

        output = np.zeros(targets.shape)

        temporal, link = targets.shape

        for time in range(temporal):
            self.step(inputs[time, :])
            output[time, :] += self.vehicle_number
        
        return mean_squared_error(targets, output)

if __name__ == "__main__":

    net = network("four_large.pkl")
    net.get_sigma_gamma(10)
    a = 1
