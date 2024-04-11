import torch
import torch.utils.data
import time
import os
import numpy as np
import csv
import dgl
from dgl.data.utils import load_graphs
import networkx as nx
from tqdm import tqdm
import random
random.seed(42)
from sklearn.model_selection import StratifiedKFold, train_test_split


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """
    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


def self_loop(g):
    """
        Utility function only, to be used only when necessary as per user self_loop flag
        : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']
        
        
        This function is called inside a function in TUsDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata['feat'] = g.ndata['feat']
    
    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    
    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata['feat'] = torch.zeros(new_g.number_of_edges())
    return new_g

name2path = {
    'abide_AAL116': '/path/to/data/abide_AAL116.bin',
    'abide_harvard48': '/path/to/data/abide_harvard48.bin',
    'abide_kmeans100': '/path/to/data/abide_kmeans100.bin',
    'abide_schaefer100': '/path/to/data/abide_schaefer100.bin',
    'abide_ward100': '/path/to/data/abide_ward100.bin',

    'adni_AAL116': '/path/to/data/adni_AAL116.bin',
    'adni_harvard48': '/path/to/data/adni_harvard48.bin',
    'adni_kmeans100': '/path/to/data/adni_kmeans100.bin',
    'adni_schaefer100': '/path/to/data/adni_schaefer100.bin',
    'adni_ward100': '/path/to/data/adni_ward100.bin',

    'neurocon_AAL116': '/path/to/data/neurocon_AAL116.bin',
    'neurocon_harvard48': '/path/to/data/neurocon_harvard48.bin',
    'neurocon_kmeans100': '/path/to/data/neurocon_kmeans100.bin',
    'neurocon_schaefer100': '/path/to/data/neurocon_schaefer100.bin',
    'neurocon_ward100': '/path/to/data/neurocon_ward100.bin',

    'ppmi_AAL116': '/path/to/data/ppmi_AAL116.bin',
    'ppmi_harvard48': '/path/to/data/ppmi_harvard48.bin',
    'ppmi_kmeans100': '/path/to/data/ppmi_kmeans100.bin',
    'ppmi_schaefer100': '/path/to/data/ppmi_schaefer100.bin',
    'ppmi_ward100': '/path/to/data/ppmi_ward100.bin',

    'taowu_AAL116': '/path/to/data/taowu_AAL116.bin',
    'taowu_harvard48': '/path/to/data/taowu_harvard48.bin',
    'taowu_kmeans100': '/path/to/data/taowu_kmeans100.bin',
    'taowu_schaefer100': '/path/to/data/taowu_schaefer100.bin',
    'taowu_ward100': '/path/to/data/taowu_ward100.bin',
}


class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, name, threshold=0.3, edge_ratio=0, node_feat_transform='original'):
        t0 = time.time()
        self.name = name

        G_dataset, Labels = load_graphs(name2path[self.name])

        self.node_num = G_dataset[0].ndata['N_features'].size(0)

        print("[!] Dataset: ", self.name)

        # transfer DGLHeteroGraph to DGLFormDataset
        data = []
        error_case = []
        for i in range(len(G_dataset)):
            if len(((G_dataset[i].ndata['N_features'] != 0).sum(dim=-1) == 0).nonzero()) > 0:
                error_case.append(i)
        print(error_case)
        G_dataset = [n for i, n in enumerate(G_dataset) if i not in error_case]

        for i in tqdm(range(len(G_dataset))):
            if edge_ratio:
                threshold_idx = int(len(G_dataset[i].edata['E_features']) * (1 - edge_ratio))
                threshold = sorted(G_dataset[i].edata['E_features'].tolist())[threshold_idx]

            G_dataset[i].remove_edges(torch.squeeze((torch.abs(G_dataset[i].edata['E_features']) < float(threshold)).nonzero()))
            G_dataset[i].edata['feat'] = G_dataset[i].edata['E_features'].unsqueeze(-1).clone()

            if name[:-7] == 'pearson' or node_feat_transform == 'original':
                G_dataset[i].ndata['feat'] = G_dataset[i].ndata['N_features'].clone()
            elif node_feat_transform == 'one_hot':
                G_dataset[i].ndata['feat'] = torch.eye(self.node_num).clone()
            elif node_feat_transform == 'pearson':
                G_dataset[i].ndata['feat'] = torch.from_numpy(np.corrcoef(G_dataset[i].ndata['N_features'].numpy())).clone()
            elif node_feat_transform == 'degree':
                G_dataset[i].ndata['feat'] = G_dataset[i].in_degrees().unsqueeze(dim=1).clone()
            elif node_feat_transform == 'adj_matrix':
                G_dataset[i].ndata['feat'] = G_dataset[i].adj().to_dense().clone()
            elif node_feat_transform == 'mean_std':
                G_dataset[i].ndata['feat'] = torch.stack(torch.std_mean(G_dataset[i].ndata['N_features'], dim=-1)).T.flip(dims=[1]).clone()
            else:
                raise NotImplementedError

            G_dataset[i].ndata.pop('N_features')
            G_dataset[i].edata.pop('E_features')
            data.append([G_dataset[i], Labels['glabel'].tolist()[i]])

        dataset = self.format_dataset(data)
        # this function splits data into train/val/test and returns the indices
        self.all_idx = self.get_all_split_idx(dataset)

        self.all = dataset
        self.train = [self.format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in range(10)]
        self.val = [self.format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in range(10)]
        self.test = [self.format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in range(10)]
        
        print("Time taken: {:.4f}s".format(time.time()-t0))

    def get_all_split_idx(self, dataset):
        """
            - Split total number of graphs into 3 (train, val and test) in 80:10:10
            - Stratified split proportionate to original distribution of data with respect to classes
            - Using sklearn to perform the split and then save the indexes
            - Preparing 10 such combinations of indexes split to be used in Graph NNs
            - As with KFold, each of the 10 fold have unique test set.
        """
        root_idx_dir = './data/{}/'.format(self.name)
        if not os.path.exists(root_idx_dir):
            os.makedirs(root_idx_dir)
        all_idx = {}

        # If there are no idx files, do the split and store the files
        if not (os.path.exists(root_idx_dir + 'train.index')):
            print("[!] Splitting the data into train/val/test ...")

            # Using 10-fold cross val to compare with benchmark papers
            k_splits = 10

            cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
            k_data_splits = []

            # this is a temporary index assignment, to be used below for val splitting
            for i in range(len(dataset.graph_lists)):
                dataset[i][0].a = lambda: None
                setattr(dataset[i][0].a, 'index', i)

            for indexes in cross_val_fold.split(dataset.graph_lists, dataset.graph_labels):
                remain_index, test_index = indexes[0], indexes[1]

                remain_set = self.format_dataset([dataset[index] for index in remain_index])

                # Gets final 'train' and 'val'
                train, val, _, __ = train_test_split(remain_set,
                                                     range(len(remain_set.graph_lists)),
                                                     test_size=0.111,
                                                     stratify=remain_set.graph_labels)

                train, val = self.format_dataset(train), self.format_dataset(val)
                test = self.format_dataset([dataset[index] for index in test_index])

                # Extracting only idx
                idx_train = [item[0].a.index for item in train]
                idx_val = [item[0].a.index for item in val]
                idx_test = [item[0].a.index for item in test]

                f_train_w = csv.writer(open(root_idx_dir + 'train.index', 'a+'))
                f_val_w = csv.writer(open(root_idx_dir + 'val.index', 'a+'))
                f_test_w = csv.writer(open(root_idx_dir + 'test.index', 'a+'))

                f_train_w.writerow(idx_train)
                f_val_w.writerow(idx_val)
                f_test_w.writerow(idx_test)

            print("[!] Splitting done!")

        # reading idx from the files
        for section in ['train', 'val', 'test']:
            with open(root_idx_dir + section + '.index', 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        return all_idx

    def format_dataset(self, dataset):  
        """
            Utility function to recover data,
            INTO-> dgl/pytorch compatible format 
        """
        graphs = [data[0] for data in dataset]
        labels = [data[1] for data in dataset]

        for graph in graphs:
            graph.ndata['feat'] = graph.ndata['feat'].float() # dgl 4.0
            # adding edge features for Residual Gated ConvNet, if not there
            if 'feat' not in graph.edata.keys():
                edge_feat_dim = graph.ndata['feat'].shape[1] # dim same as node feature dim
                graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)

        return DGLFormDataset(graphs, labels)
    
    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        batched_graph = dgl.batch(graphs)
        
        return batched_graph, labels
    
    # prepare dense tensors for GNNs using them; such as RingGNN, 3WLGNN
    def collate_dense_gnn(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
    
        g = graphs[0]
        adj = self._sym_normalize_adj(g.adjacency_matrix().to_dense())        
        """
            Adapted from https://github.com/leichen2018/Ring-GNN/
            Assigning node and edge feats::
            we have the adjacency matrix in R^{n x n}, the node features in R^{d_n} and edge features R^{d_e}.
            Then we build a zero-initialized tensor, say T, in R^{(1 + d_n + d_e) x n x n}. T[0, :, :] is the adjacency matrix.
            The diagonal T[1:1+d_n, i, i], i = 0 to n-1, store the node feature of node i. 
            The off diagonal T[1+d_n:, i, j] store edge features of edge(i, j).
        """

        zero_adj = torch.zeros_like(adj)
        
        in_dim = g.ndata['feat'].shape[1]
        
        # use node feats to prepare adj
        adj_node_feat = torch.stack([zero_adj for j in range(in_dim)])
        adj_node_feat = torch.cat([adj.unsqueeze(0), adj_node_feat], dim=0)
        
        for node, node_feat in enumerate(g.ndata['feat']):
            adj_node_feat[1:, node, node] = node_feat

        x_node_feat = adj_node_feat.unsqueeze(0)
        
        return x_node_feat, labels
    
    def _sym_normalize_adj(self, adj):
        deg = torch.sum(adj, dim=0)
        deg_inv = torch.where(deg>0, 1./torch.sqrt(deg), torch.zeros(deg.size()))
        deg_inv = torch.diag(deg_inv)
        return torch.mm(deg_inv, torch.mm(adj, deg_inv))

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True
        for split_num in range(10):
            self.train[split_num].graph_lists = [self_loop(g) for g in self.train[split_num].graph_lists]
            self.val[split_num].graph_lists = [self_loop(g) for g in self.val[split_num].graph_lists]
            self.test[split_num].graph_lists = [self_loop(g) for g in self.test[split_num].graph_lists]
            
        for split_num in range(10):
            self.train[split_num] = DGLFormDataset(self.train[split_num].graph_lists, self.train[split_num].graph_labels)
            self.val[split_num] = DGLFormDataset(self.val[split_num].graph_lists, self.val[split_num].graph_labels)
            self.test[split_num] = DGLFormDataset(self.test[split_num].graph_lists, self.test[split_num].graph_labels)
