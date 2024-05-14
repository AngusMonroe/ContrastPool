import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from scipy.linalg import block_diag
from torch.autograd import Function
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


class ContrastPoolLayer(nn.Module):

    def __init__(self, input_dim, assign_dim, output_feat_dim,
                 activation, dropout, aggregator_type, link_pred, batch_norm, pool_assign='GraphSage', max_node_num=0):
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

        # cs
        self.weight = nn.Parameter(torch.Tensor(max_node_num, assign_dim))
        self.bias = nn.Parameter(torch.Tensor(1, assign_dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h, diff_h=None, adj=None, e=None):
        # h: [1000, 86]
        batch_size = len(g.batch_num_nodes())
        feat, e = self.feat_gc(g, h, e)
        device = feat.device
        # GCN
        if diff_h is not None:
            # print(diff_h.shape)
            # print(self.weight.shape)
            support = torch.matmul(diff_h, self.weight)
            if adj is not None:
                output = torch.matmul(adj.to(device), support)
            else:
                output = torch.matmul(g.adj().to_dense().clone().to(device), support.repeat(batch_size, 1))
            assign_tensor = output + self.bias
        else:
            assign_tensor, e = self.pool_gc(g, h, e)
        # assign_tensor: [2000, 50]
        # print(assign_tensor.shape)

        assign_tensor_masks = []
        assign_size = int(assign_tensor.size()[1]) if adj is not None else int(assign_tensor.size()[1] / batch_size)
        for g_n_nodes in g.batch_num_nodes():
            mask = torch.ones((g_n_nodes, assign_size))
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
        if adj is not None:
            assign_tensor = assign_tensor.repeat(batch_size, batch_size)

        assign_tensor = masked_softmax(assign_tensor, mask, memory_efficient=False)
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
        self.link_pred_layer = LinkPredLoss()
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
