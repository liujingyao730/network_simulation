import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch

def sparselist_to_tensor(adj_list):

    tensor = np.array(adj_list[0].todense())[None, :, :]
    for i in range(len(adj_list)-1):
        tensor = np.concatenate((tensor, np.array(adj_list[i+1].todense())[None, :, :]), axis=0)

    return tensor

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def adj_to_laplace(adj_list):

    D_tilde = torch.diag_embed(torch.pow(torch.sum(adj_list, dim=1), -1 / 2))
    laplace = torch.bmm(D_tilde, adj_list)
    laplace = torch.bmm(laplace, D_tilde)

    return laplace

def show_heat(network_layout, input_data, file="heat_map"):
        
    assert len(network_layout) == input_data.shape[0]

    x = np.array([network_layout[i][0] for i in range(len(network_layout))])
    y = np.array([network_layout[i][1] for i in range(len(network_layout))])
    input_data = np.array(input_data)

    x_min = x.min()
    x_max = x.max()
    y_min = y.min()
    y_max = y.max()

    x_t = np.linspace(x_min, x_max, 100)
    y_t = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x_t, y_t)

    interpolant = interp2d(x, y, input_data, kind="linear")

    plt.figure()
    plt.axes().set_aspect("equal")
    plt.pcolor(X, Y, interpolant(x_t, y_t))
    plt.scatter(x, y, 25, input_data)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.savefig(file)

def from_sparse_get_index(adj_list):

    time = len(adj_list)
    N = adj_list[0].shape[0]
    index = np.zeros((time, N, 2)).astype('int') - 1
    weight = np.zeros((time, N, 2))
    
    for t in range(time):
        adj_t = adj_list[t]
        adj = sp.coo_matrix(adj_t)
        for i in range(len(adj.row)):
            r = adj.row[i]
            c = adj.col[i]
            if index[t, r, 0] > -1:
                index[t, r, 1] = c
                weight[t, r, 1] = 1
            else:
                index[t, r, 0] = c
                weight[t, r, 0] = 1
    
    return index, weight
