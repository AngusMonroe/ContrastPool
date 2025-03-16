# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import networkx as nx
import os  # To create directories
import shutil
import scipy.io
import dgl
import torch
import glob
import csv
import re
import json
from tqdm import tqdm
from dgl.data.utils import save_graphs
from sklearn.model_selection import StratifiedKFold, train_test_split


def _load_matrix_subject_with_files(files, remove_negative=False):
    subjects = []
    for file in files:
        mat = scipy.io.loadmat(file)
        mat = mat["data"]
        np.fill_diagonal(mat, 0)
        if remove_negative:
            mat[mat < 0] = 0
        subjects.append(mat)
    return np.array(subjects)

def construct_dataset(data_name):
    feat_dir = 'data/to/connectivity_matrices_schaefer/' + data_name + '/'

    G_dataset = []
    Labels = []
    group2idx = {}
    paths = glob.glob(feat_dir + '/*/' + '*_features_timeseries.mat', recursive=True)
    feats = _load_matrix_subject_with_files(paths)

    print('Processing ' + data_name + '...')

    for j in tqdm(range(len(feats))):
        name = paths[j].split('/')[-1]
        group = re.findall('sub-([^\d]+)', name)[0]
        if group not in group2idx.keys():
            group2idx[group] = len(group2idx.keys())
        i = group2idx[group]

        G = nx.DiGraph(np.ones([feats[j].shape[0], feats[j].shape[0]]))
        graph_dgl = dgl.from_networkx(G)

        graph_dgl.ndata['N_features'] = torch.from_numpy(feats[j])
        # Include edge features
        weights = []
        for u, v, w in G.edges.data('weight'):
            # if w is not None:
            weights.append(w)
        graph_dgl.edata['E_features'] = torch.Tensor(weights)

        G_dataset.append(graph_dgl)
        Labels.append(i)

    print('Finish process ' + data_name + '. ' + str(len(feats)) + ' subjects in total.')

    Labels = torch.LongTensor(Labels)
    graph_labels = {"glabel": Labels}
    if not os.path.exists('./bin_dataset/'):
        os.mkdir('./bin_dataset/')
    print(Labels.shape)
    print(len(G_dataset))
    save_graphs("./bin_dataset/" + data_name + ".bin", G_dataset, graph_labels)


def move_files(data_name):
    feat_dir = '/data/jiaxing/brain/connectivity_matrices_schaefer/' + data_name + '/'
    paths = glob.glob(feat_dir + '/*/*', recursive=True)
    for path in paths:
        if path[-4:] == '.mat':
            if 'schashaefer' in path:
                new_path = re.sub('schashaefer', 'schaefer', path)
                os.rename(path, new_path)
            continue
        else:
            parcellation = data_name.split('_')[-1]
            os.rename(path, path + '_' + parcellation + '_correlation_matrix.mat')


if __name__ == '__main__':
    error_name = []
    # file_name_list = os.listdir('./correlation_datasets/')
    file_name_list = ['adni_schaefer100']

    for data_name in file_name_list:
        move_files(data_name)
        # construct_dataset(data_name)
        # try:
        #     construct_dataset(data_name)
        # except:
        #     print('[ERROR]: ' + data_name)
        #     error_name.append(data_name)
    print(error_name)
    print('Done!')
