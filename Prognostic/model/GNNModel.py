import os
import torch.nn.functional as F


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model.layers import IBN
from torch_geometric.nn import global_mean_pool, global_max_pool, global_sort_pool, dense_diff_pool, global_add_pool, \
    TopKPooling, ASAPooling, SAGPooling
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GraphConv, LEConv, LayerNorm, GATConv, TransformerConv
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils.block_utils import Block, weights_init_kaiming, weights_init_classifier


class Intra_GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, drop_out_ratio=0.2,
                 num_heads=4, use_norm=False):
        super(Intra_GCN, self).__init__()
        self.use_norm = use_norm

        self.conv1 = LEConv(in_channels=in_feats, out_channels=out_feats)
        self.conv2 = SAGEConv(n_hidden, n_hidden)
        self.conv3 = GATConv(n_hidden, n_hidden)
        self.conv4 = SAGEConv(n_hidden, out_feats)
        self.norm = torch.nn.BatchNorm1d(in_feats)
        # self.norm1 = LayerNorm(n_hidden)
        # for convs in [self.conv1, self.conv2, self.conv3, self.conv4]:
        #     convs.reset_parameters()
        self.drop_out_ratio = drop_out_ratio

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        x = self.norm(x)
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = F.dropout(x, p=self.drop_out_ratio, training=self.training)
        fea = x

        return fea, x

class Res_GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, drop_out_ratio=0.2,
                 gnn_method=['sage'], num_heads=2, use_gnn_norm=False):
        super(Res_GCN, self).__init__()

        self.use_gnn_norm = use_gnn_norm
        self.gnn_method = gnn_method
        self.num_layers = len(gnn_method)
        self.gcnlayers = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        GraphConv = lambda i, h: GINConv(nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)))


        for layer in range(self.num_layers):
            if layer == 0:
                input_hidden = in_feats
                output_hidden = n_hidden
            elif layer+1 <= self.num_layers:
                input_hidden = n_hidden
                output_hidden = n_hidden
            else:
                input_hidden = n_hidden
                output_hidden = out_feats

            if gnn_method[layer] == 'sage':
                conv = SAGEConv(input_hidden, output_hidden)
                conv.reset_parameters()
            elif gnn_method[layer] == 'gin':
                conv = GraphConv(input_hidden, output_hidden)
                conv.reset_parameters()
            elif gnn_method[layer] == 'gcn':
                conv = GCNConv(in_channels=input_hidden, out_channels=output_hidden)
                # conv.reset_parameters()
            elif gnn_method[layer] == 'gat':
                conv = GATConv(in_channels=input_hidden, out_channels=output_hidden, heads=num_heads, concat=False)
                conv.reset_parameters()
            elif gnn_method[layer] == 'leconv':
                conv = LEConv(in_channels=input_hidden, out_channels=output_hidden)
                # conv.reset_parameters()
            elif gnn_method[layer] == 'graphconv':
                conv = GraphConv(in_channels=input_hidden, out_channels=output_hidden)
                # conv.reset_parameters()
            elif gnn_method[layer] == 'TransformerConv':
                conv = TransformerConv(input_hidden, output_hidden, heads=num_heads, beta=True, concat=False)
                # conv.reset_parameters()
            else:
                raise NotImplementedError

            self.gcnlayers.append(conv)

        if self.use_gnn_norm:
            self.norms = torch.nn.ModuleList()
            for layer in range(self.num_layers-1):
                self.norms.append(IBN(n_hidden))
        self.drop_out_ratio = drop_out_ratio

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        mid_node_feats = []
        for i in range(self.num_layers - 2):
            if self.gnn_method[i] in ['gcn', 'leconv', 'graphconv']:
                x = self.gcnlayers[i](x, edge_index, edge_weight=edge_weight)
            else:
                x = self.gcnlayers[i](x, edge_index)

            if self.use_gnn_norm:
                x = self.norms[i](x)
            x = F.relu(x)

            # if i>=2:
            mid_node_feats.append(x)

        x = self.gcnlayers[self.num_layers-1](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_out_ratio, training=self.training)
        fea = x

        return fea, mid_node_feats

class GNNModel(nn.Module):
    def __init__(self, in_feats_intra=1024, n_hidden_intra=1024,
                 out_feats_intra=1024, domain_nums=6,
                 gnn_intra=['sage'], num_heads=1, mpool_inter="mean", use_gnn_norm=False):
        super(GNNModel, self).__init__()

        self.mpool_inter = mpool_inter
        # # intra-graph
        self.gcn1 = Res_GCN(
            in_feats=in_feats_intra, n_hidden=n_hidden_intra,
            out_feats=out_feats_intra,
            gnn_method=gnn_intra,
            num_heads=num_heads,
            use_gnn_norm=use_gnn_norm
        )

        self.os_head = nn.Linear(out_feats_intra, 1, bias=False)
        self.timefit_head = nn.Linear(out_feats_intra, 1, bias=False)
        self.gcls_head = nn.Linear(out_feats_intra, domain_nums, bias=False)
        # self.os_head.apply(weights_init_classifier)

    def forward(self, batch_pyg, domain_labels=None):
        n_feat, mid_node_feats = self.gcn1(batch_pyg)
        mid_g_feats = []
        if self.mpool_inter == 'mean':
            g_feat = global_mean_pool(n_feat, batch_pyg.batch)
            for inode in mid_node_feats:
                mid_g_feats.append(global_mean_pool(inode, batch_pyg.batch))
        elif self.mpool_inter == 'max':
            g_feat = global_max_pool(n_feat, batch_pyg.batch)
            for inode in mid_node_feats:
                mid_g_feats.append(global_max_pool(inode, batch_pyg.batch))
        elif self.mpool_inter == 'sum':
            g_feat = global_add_pool(n_feat, batch_pyg.batch)
            for inode in mid_node_feats:
                mid_g_feats.append(global_add_pool(inode, batch_pyg.batch))
        # print(self.os_head(g_feat))
        os_time = self.os_head(g_feat)
        g_id = self.gcls_head(g_feat)
        time_fit = self.timefit_head(g_feat)
        return {
            'node_feats': n_feat,
            'mid_node_feats': mid_node_feats,
            'mid_g_feats': mid_g_feats,
            'g_feat': g_feat,
            'mid_g_feat': g_feat,
            'os_time': os_time,
            'g_id': g_id,
            'time_fit': time_fit,
        }