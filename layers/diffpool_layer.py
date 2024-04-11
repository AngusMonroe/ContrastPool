import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from scipy.linalg import block_diag

from torch.autograd import Function

"""
    DIFFPOOL:
    Z. Ying, J. You, C. Morris, X. Ren, W. Hamilton, and J. Leskovec, 
    Hierarchical graph representation learning with differentiable pooling (NeurIPS 2018)
    https://arxiv.org/pdf/1806.08804.pdf
    
    ! code started from dgl diffpool examples dir
"""

from layers.graphsage_layer import GraphSageLayer, DenseGraphSage


def masked_softmax(matrix, mask, dim=-1, memory_efficient=True,
                   mask_fill_value=-1e32):
    '''
    masked_softmax for dgl batch graph
    code snippet contributed by AllenNLP (https://github.com/allenai/allennlp)
    '''
    if mask is None:
        result = torch.nn.functional.softmax(matrix, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < matrix.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(matrix * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_matrix = matrix.masked_fill((1 - mask).byte(),
                                               mask_fill_value)
            result = torch.nn.functional.softmax(masked_matrix, dim=dim)
    return result


class EntropyLoss(nn.Module):
    # Return Scalar
    # loss used in diffpool
    def forward(self, adj, anext, s_l):
        entropy = (torch.distributions.Categorical(
            probs=s_l).entropy()).sum(-1).mean(-1)
        assert not torch.isnan(entropy)
        return entropy


class DiffPoolLayer(nn.Module):

    def __init__(self, input_dim, assign_dim, output_feat_dim,
                 activation, dropout, aggregator_type, link_pred, batch_norm, pool_assign='GraphSage'):
        super().__init__()
        self.embedding_dim = input_dim
        self.assign_dim = assign_dim
        self.hidden_dim = output_feat_dim
        self.link_pred = link_pred
        self.feat_gc = GraphSageLayer(
            input_dim,
            output_feat_dim,
            activation,
            dropout,
            aggregator_type,
            batch_norm)
        if pool_assign == 'GraphSage':
            self.pool_gc = GraphSageLayer(
                input_dim,
                assign_dim,
                activation,
                dropout,
                aggregator_type,
                batch_norm)
        else:
            pass
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        self.reg_loss.append(EntropyLoss())

    def forward(self, g, h, e=None):
        # h: [1000, 86]
        feat, e = self.feat_gc(g, h, e)
        device = feat.device
        assign_tensor, e = self.pool_gc(g, h, e)

        assign_tensor_masks = []
        batch_size = len(g.batch_num_nodes())
        for g_n_nodes in g.batch_num_nodes():
            mask = torch.ones((g_n_nodes,
                               int(assign_tensor.size()[1] / batch_size)))
            assign_tensor_masks.append(mask)
        """
        The first pooling layer is computed on batched graph.
        We first take the adjacency matrix of the batched graph, which is block-wise diagonal.
        We then compute the assignment matrix for the whole batch graph, which will also be block diagonal
        """
        mask = torch.FloatTensor(
            block_diag(
                *
                assign_tensor_masks)).to(
            device=device)

        assign_tensor = masked_softmax(assign_tensor, mask,
                                       memory_efficient=False)
        # print(assign_tensor.shape)
        h = torch.matmul(torch.t(assign_tensor), feat)                     # equation (3) of DIFFPOOL paper
        adj = g.adjacency_matrix(ctx=device)

        adj_new = torch.sparse.mm(adj, assign_tensor)
        adj_new = torch.mm(torch.t(assign_tensor), adj_new)                # equation (4) of DIFFPOOL paper

        if self.link_pred:
            current_lp_loss = torch.norm(adj.to_dense() -
                                         torch.mm(assign_tensor, torch.t(assign_tensor))) / np.power(g.number_of_nodes(), 2)
            self.loss_log['LinkPredLoss'] = current_lp_loss

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)

            self.loss_log[loss_name] = loss_layer(adj, adj_new, assign_tensor)
        return adj_new, h


class LinkPredLoss(nn.Module):
    # loss used in diffpool
    def forward(self, adj, anext, s_l):
        link_pred_loss = (
                adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()


class DenseDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, link_pred=False, entropy=True):
        super().__init__()
        self.link_pred = link_pred
        self.log = {}
        self.link_pred_layer = self.LinkPredLoss()
        self.embed = DenseGraphSage(nfeat, nhid, use_bn=True)
        self.assign = DiffPoolAssignment(nfeat, nnext)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        if link_pred:
            self.reg_loss.append(LinkPredLoss())
        if entropy:
            self.reg_loss.append(EntropyLoss())

    def forward(self, x, adj, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, anext, s_l)
        if log:
            self.log['a'] = anext.cpu().numpy()
        return xnext, anext


class DiffPoolAssignment(nn.Module):
    def __init__(self, nfeat, nnext):
        super().__init__()
        self.assign_mat = DenseGraphSage(nfeat, nnext, use_bn=True)

    def forward(self, x, adj, log=False):
        s_l_init = self.assign_mat(x, adj)
        s_l = F.softmax(s_l_init, dim=-1)
        return s_l
