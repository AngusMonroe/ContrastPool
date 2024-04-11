import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from layers.attention_layer import EncoderLayer
import time
import numpy as np
from scipy.linalg import block_diag
import dgl

from layers.graphsage_layer import GraphSageLayer, DenseGraphSage
from layers.contrastpool_layer import ContrastPoolLayer, DenseDiffPool


class ContrastPoolNet(nn.Module):
    """
    DiffPool Fuse with GNN layers and pooling layers in sequence
    """

    def __init__(self, net_params, pool_ratio=0.5):

        super().__init__()
        input_dim = net_params['in_dim']
        self.hidden_dim = net_params['hidden_dim']
        embedding_dim = net_params['hidden_dim'] 
        out_dim = net_params['hidden_dim']
        self.n_classes = net_params['n_classes']
        activation = F.relu
        n_layers = net_params['L'] 
        dropout = net_params['dropout']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        aggregator_type = net_params['sage_aggregator']
        self.lambda1 = net_params['lambda1']
        self.learnable_q = net_params['learnable_q']

        self.device = net_params['device']
        self.link_pred = True  
        self.concat = False  
        self.n_pooling = 1  
        self.batch_size = net_params['batch_size']
        if 'pool_ratio' in net_params.keys():
            pool_ratio = net_params['pool_ratio']
        self.e_feat = net_params['edge_feat']
        self.link_pred_loss = []
        self.entropy_loss = []

        self.embedding_h = nn.Linear(input_dim, self.hidden_dim)

        # list of GNN modules before the first diffpool operation
        self.gc_before_pool = nn.ModuleList()

        self.assign_dim = int(net_params['max_num_node'] * pool_ratio) 
        self.bn = True
        self.num_aggs = 1

        # constructing layers
        # layers before diffpool
        assert n_layers >= 2, "n_layers too few"
        self.gc_before_pool.append(GraphSageLayer(self.hidden_dim, self.hidden_dim, activation,
                                                  dropout, aggregator_type, self.residual, self.bn, e_feat=self.e_feat))

        for _ in range(n_layers - 2):
            self.gc_before_pool.append(GraphSageLayer(self.hidden_dim, self.hidden_dim, activation,
                                                      dropout, aggregator_type, self.residual, self.bn, e_feat=self.e_feat))

        self.gc_before_pool.append(GraphSageLayer(self.hidden_dim, embedding_dim, None, dropout, aggregator_type, self.residual, e_feat=self.e_feat))


        assign_dims = []
        assign_dims.append(self.assign_dim)
        if self.concat:
            # diffpool layer receive pool_emedding_dim node feature tensor
            # and return pool_embedding_dim node embedding
            pool_embedding_dim = self.hidden_dim * (n_layers - 1) + embedding_dim
        else:

            pool_embedding_dim = embedding_dim

        self.first_diffpool_layer = ContrastPoolLayer(pool_embedding_dim, self.assign_dim, self.hidden_dim, activation,
                                                dropout, aggregator_type, self.link_pred, self.batch_norm,
                                                max_node_num=net_params['max_num_node'])
        gc_after_per_pool = nn.ModuleList()

        # list of list of GNN modules, each list after one diffpool operation
        self.gc_after_pool = nn.ModuleList()

        for _ in range(n_layers - 1):
            gc_after_per_pool.append(DenseGraphSage(self.hidden_dim, self.hidden_dim, self.residual))
        gc_after_per_pool.append(DenseGraphSage(self.hidden_dim, embedding_dim, self.residual))
        self.gc_after_pool.append(gc_after_per_pool)

        self.assign_dim = int(self.assign_dim * pool_ratio)

        self.diffpool_layers = nn.ModuleList()
        # each pooling module
        for _ in range(self.n_pooling - 1):
            self.diffpool_layers.append(DenseDiffPool(pool_embedding_dim, self.assign_dim, self.hidden_dim, self.link_pred))

            gc_after_per_pool = nn.ModuleList()

            for _ in range(n_layers - 1):
                gc_after_per_pool.append(DenseGraphSage(self.hidden_dim, self.hidden_dim, self.residual))
            gc_after_per_pool.append(DenseGraphSage(self.hidden_dim, embedding_dim, self.residual))
            self.gc_after_pool.append(gc_after_per_pool)

            assign_dims.append(self.assign_dim)
            self.assign_dim = int(self.assign_dim * pool_ratio)

        # predicting layer
        if self.concat:
            self.pred_input_dim = pool_embedding_dim * \
                                  self.num_aggs * (self.n_pooling + 1)
        else:
            self.pred_input_dim = embedding_dim * self.num_aggs
        self.pred_layer = nn.Linear(self.pred_input_dim, self.n_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

        self.contrast_adj = None
        self.adj_dict = None
        self.nodes_dict = None
        self.nodes1 = None
        self.nodes2 = None
        self.encoder1 = None
        self.encoder2 = None
        self.encoder1_node = None
        self.encoder2_node = None
        self.num_A = None
        self.num_B = None
        self.node_num = None
        self.diff_h = None
        self.attn_loss = None
        self.ad_adj = None
        self.softmax = nn.Softmax(dim=-1)
        # self.sim = nn.CosineSimilarity(dim=-1, eps=1e-08)

    def cal_attn_loss(self, attn):
        entropy = (torch.distributions.Categorical(logits=attn).entropy()).mean()
        assert not torch.isnan(entropy)
        return entropy

    def cal_contrast(self, trainset, device, merge_classes=True):
        from contrast_subgraph import get_summary_tensor
        G_dataset = trainset[:][0]
        Labels = torch.tensor(trainset[:][1])

        self.adj_dict, self.nodes_dict = get_summary_tensor(G_dataset, Labels, device, merge_classes=merge_classes)
        self.node_num = G_dataset[0].ndata['feat'].size(0)
        feat_dim = G_dataset[0].ndata['feat'].size(1)

        learnable_q = self.learnable_q
        n_head = 1
        self.encoder1 = EncoderLayer(self.node_num, n_head, self.node_num, 0.0, device, self.node_num, learnable_q, pos_enc='index').to(device)
        self.encoder2 = EncoderLayer(self.node_num, n_head, self.node_num, 0.0, device, self.node_num, learnable_q).to(device)
        self.encoder1_node = EncoderLayer(self.node_num, n_head, self.node_num, 0.0, device, feat_dim, learnable_q, pos_enc='index').to(device)
        self.encoder2_node = EncoderLayer(self.node_num, n_head, self.node_num, 0.0, device, feat_dim, learnable_q).to(device)

    def cal_contrast_adj(self, device):
        adj_list = []
        nodes_list = []
        for i in self.adj_dict.keys():
            adj = self.encoder1(self.adj_dict[i])
            adj = self.encoder2(adj.permute(1, 0, 2))
            adj_list.append(adj.mean(1))

            nodes_feat = self.encoder1_node(self.nodes_dict[i])
            nodes_feat = self.encoder2_node(nodes_feat.permute(1, 0, 2))
            nodes_list.append(nodes_feat.mean(1))
        self.ad_adj = torch.stack(adj_list)
        adj_var = torch.std(torch.stack(adj_list).to(device), 0)
        nodes_var = torch.std(torch.stack(nodes_list).to(device), 0)

        self.contrast_adj = adj_var
        self.diff_h = nodes_var
        self.attn_loss = self.cal_attn_loss(self.contrast_adj)

        self.contrast_adj_trans = self.contrast_adj

    def gcn_forward(self, g, h, e, gc_layers, cat=False):
        """
        Return gc_layer embedding cat.
        """
        block_readout = []
        for gc_layer in gc_layers[:-1]:
            h, e = gc_layer(g, h, e)
            block_readout.append(h)
        h, e = gc_layers[-1](g, h, e)
        block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=1)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def gcn_forward_tensorized(self, h, adj, gc_layers, cat=False):
        block_readout = []
        for gc_layer in gc_layers:
            h = gc_layer(h, adj)
            block_readout.append(h)
        if cat:
            block = torch.cat(block_readout, dim=2)  # N x F, F = F1 + F2 + ...
        else:
            block = h
        return block

    def forward(self, g, h, e):
        self.link_pred_loss = []
        self.entropy_loss = []

        # node feature for assignment matrix computation is the same as the
        # original node feature
        h = self.embedding_h(h)

        out_all = []

        # we use GCN blocks to get an embedding first
        g_embedding = self.gcn_forward(g, h, e, self.gc_before_pool, self.concat)

        g.ndata['h'] = g_embedding

        readout = dgl.sum_nodes(g, 'h')
        out_all.append(readout)
        if self.num_aggs == 2:
            readout = dgl.max_nodes(g, 'h')
            out_all.append(readout)

        self.cal_contrast_adj(device=h.device)
        adj, h = self.first_diffpool_layer(g, g_embedding, self.diff_h, self.contrast_adj_trans)
        node_per_pool_graph = int(adj.size()[0] / self.batch_size)

        h, adj = self.batch2tensor(adj, h, node_per_pool_graph)
        h = self.gcn_forward_tensorized(h, adj, self.gc_after_pool[0], self.concat)

        readout = torch.sum(h, dim=1)
        out_all.append(readout)
        if self.num_aggs == 2:
            readout, _ = torch.max(h, dim=1)
            out_all.append(readout)

        for i, diffpool_layer in enumerate(self.diffpool_layers):
            h, adj = diffpool_layer(h, adj)
            h = self.gcn_forward_tensorized(h, adj, self.gc_after_pool[i + 1], self.concat)

            readout = torch.sum(h, dim=1)
            out_all.append(readout)

            if self.num_aggs == 2:
                readout, _ = torch.max(h, dim=1)
                out_all.append(readout)

        if self.concat or self.num_aggs > 1:
            hg = torch.cat(out_all, dim=1)
        else:
            hg = readout

        ypred = self.pred_layer(hg)
        return ypred

    def batch2tensor(self, batch_adj, batch_feat, node_per_pool_graph):
        """
        transform a batched graph to batched adjacency tensor and node feature tensor
        """
        batch_size = int(batch_adj.size()[0] / node_per_pool_graph)
        adj_list = []
        feat_list = []

        for i in range(batch_size):
            start = i * node_per_pool_graph
            end = (i + 1) * node_per_pool_graph

            # 1/sqrt(V) normalization
            snorm_n = torch.FloatTensor(node_per_pool_graph, 1).fill_(1./float(node_per_pool_graph)).sqrt().to(self.device)

            adj_list.append(batch_adj[start:end, start:end])
            feat_list.append((batch_feat[start:end, :])*snorm_n)
        adj_list = list(map(lambda x: torch.unsqueeze(x, 0), adj_list))
        feat_list = list(map(lambda x: torch.unsqueeze(x, 0), feat_list))
        adj = torch.cat(adj_list, dim=0)
        feat = torch.cat(feat_list, dim=0)

        return feat, adj

    def loss(self, pred, label):
        '''
        loss function
        '''
        #softmax + CE
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        e1_loss = 0.0
        for diffpool_layer in self.diffpool_layers:
            for key, value in diffpool_layer.loss_log.items():
                e1_loss += value
        loss += e1_loss + self.lambda1 * self.attn_loss
        return loss
