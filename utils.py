import numpy as np
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
    laplace = torch.einsum("tbc,tcd->tbd", D_tilde, adj_list)
    laplace = torch.einsum("tbc,tcd->tbd", laplace, D_tilde)

    return laplace
