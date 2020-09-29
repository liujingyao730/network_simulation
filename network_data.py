import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
from tqdm import tqdm

class network_dataset(Dataset):

    def __init__(self, args, cell_list):

        super().__init__()

        self.sim_step = args["sim_step"]
        self.delta_T = args["delta_T"]
        self.temporal_length = args["temporal_length"]
        self.init_length = args["init_length"]
        self.prefix = args["prefix"]
        self.data_fold = args["data_fold"]

        self.load_data()
        #self.normalize()

    def load_data(self):

        self.dest1_file = os.path.join(self.data_fold, self.prefix+"_dest1.csv")
        self.dest2_file = os.path.join(self.data_fold, self.prefix+"_dest2.csv")
        self.dest3_file = os.path.join(self.data_fold, self.prefix+"_dest3.csv")
        self.dest4_file = os.path.join(self.data_fold, self.prefix+"_dest4.csv")
        self.dest5_file = os.path.join(self.data_fold, self.prefix+"_dest5.csv")
        self.dest6_file = os.path.join(self.data_fold, self.prefix+"_dest6.csv")

        self.dest1 = pd.read_csv(self.dest1_file, index_col=0)[cell_list].values[:, :, np.newaxis]
        self.dest2 = pd.read_csv(self.dest2_file, index_col=0)[cell_list].values[:, :, np.newaxis]
        self.dest3 = pd.read_csv(self.dest3_file, index_col=0)[cell_list].values[:, :, np.newaxis]
        self.dest4 = pd.read_csv(self.dest4_file, index_col=0)[cell_list].values[:, :, np.newaxis]
        self.dest5 = pd.read_csv(self.dest5_file, index_col=0)[cell_list].values[:, :, np.newaxis]
        self.dest6 = pd.read_csv(self.dest6_file, index_col=0)[cell_list].values[:, :, np.newaxis]

        self.data = np.concatenate((self.dest1, self.dest2, self.dest3, self.dest4, self.dest5, self.dest6), axis=2)
        self.time_number = int(self.data.shape[0] - (self.temporal_length + 1) * self.delta_T / self.sim_step)
    
    def normalize(self):
        try:
            mean = np.mean(self.data, axis=2)
            std = np.std(self.data, axis=2)
            self.data = self.data - mean[:, :, None]
            self.data = self.data / std[:, :, None]
        except AttributeError :
            print("not load data yet")
            os._exit(1)

    def __getitem__(self, index):

        input_time = [int(i*self.delta_T/self.sim_step) + index for i in range(self.temporal_length)]
        output_time = [int(i*self.delta_T/self.sim_step) + index for i in range(self.init_length, self.temporal_length+1)]
        input_tensor = self.data[input_time]
        output_tensor = self.data[output_time]

        return (input_tensor, output_tensor)

    def __len__(self):
        
        return self.time_number


if __name__ == "__main__":   
    args = {}
    args["sim_step"] = 0.1
    args["delta_T"] = 5
    args["temporal_length"] = 16
    args["init_length"] = 4
    args["prefix"] = "default"
    args["data_fold"] = "data"
    cell_list = ['gneJ2', 'gneJ1', '-gneE0-0', '-gneE0-1', '-gneE0-2', '-gneE1-0']
    data_set = network_dataset(args, cell_list)
    dataloader = torch.utils.data.DataLoader(data_set, batch_size=10)
    for i, data in tqdm(enumerate(dataloader)):
        print(i)
        a = 1