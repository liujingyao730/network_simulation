import numpy as np

def sparselist_to_tensor(adj_list):

    tensor = np.array(adj_list[0].todense())[None, :, :]
    for i in range(len(adj_list)-1):
        tensor = np.concatenate((tensor, np.array(adj_list[i+1].todense())[None, :, :]), axis=0)

    return tensor

