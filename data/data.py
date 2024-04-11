"""
    File to load dataset based on user control from main file
"""
from data.BrainNet import BrainDataset


def LoadData(DATASET_NAME, threshold=0, edge_ratio=0, node_feat_transform='original'):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """

    return BrainDataset(DATASET_NAME, threshold=threshold, edge_ratio=edge_ratio, node_feat_transform=node_feat_transform)
