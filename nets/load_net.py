"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.contrastpool_net import ContrastPoolNet


def ContrastPool(net_params):
    return ContrastPoolNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        "ContrastPool": ContrastPool
    }
    model = models[MODEL_NAME](net_params)
    model.name = MODEL_NAME
        
    return model
