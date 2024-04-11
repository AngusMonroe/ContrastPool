import heapq
import math
import numpy as np
import torch
import dgl
from dgl.data.utils import load_graphs
from copy import deepcopy
from tqdm import tqdm


def get_summary_tensor(G_dataset, Labels, device, merge_classes=False):
    num_G = len(G_dataset)
    Labels = Labels.tolist()
    node_num = G_dataset[0].ndata['feat'].shape[0]
    adj_dict = {}
    nodes_dict = {}
    final_adj_dict = {}
    final_nodes_dict = {}
    for i in range(num_G):
        if Labels[i] not in adj_dict.keys():
            adj_dict[Labels[i]] = []
            nodes_dict[Labels[i]] = []
        adj_dict[Labels[i]].append(G_dataset[i].edata['feat'].squeeze().view(node_num, -1).tolist())
        nodes_dict[Labels[i]].append(G_dataset[i].ndata['feat'].tolist())

    for i in adj_dict.keys():
        final_adj_dict[i] = torch.tensor(adj_dict[i]).to(device)
        final_nodes_dict[i] = torch.tensor(nodes_dict[i]).to(device)
    return final_adj_dict, final_nodes_dict
