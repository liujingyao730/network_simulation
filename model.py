import torch
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv
from torch.nn import functional as F
import torch.nn as nn
import os
import pickle
import numpy as np 
import pandas as pd 
import math
import random
from torch.autograd import Variable
from torch.utils.data import Dataset

class test_model(nn.Module):

    def __init__(self, args):

        super().__init__()
        self.init_length = args["init_length"]

    def forward(self, input_data):

        return input_data[:, self.init_length:-1, :, :]

    def infer(self, input_data, cell_list):

        return input_data[:, self.init_length:-1, :, :]