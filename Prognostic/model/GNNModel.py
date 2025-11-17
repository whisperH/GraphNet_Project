import torch.nn.functional as F
from Prognostic.model.layers import IBN
from torch_geometric.nn import global_mean_pool, global_max_pool, global_sort_pool, dense_diff_pool, global_add_pool, \
    TopKPooling, ASAPooling, SAGPooling
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, LEConv, BatchNorm, GATConv, TransformerConv, LayerNorm
import torch
import torch.nn as nn

class Res_GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, drop_out_ratio=0.2,
                 gnn_method=['sage'], num_heads=2, use_gnn_norm=False, IN_Ratio=0.5):
        super(Res_GCN, self).__init__()

        self.IN_Ratio = IN_Ratio
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
            elif layer+1 < self.num_layers:
                input_hidden = n_hidden
                output_hidden = n_hidden
            else:
                input_hidden = n_hidden
                output_hidden = out_feats

            if gnn_method[layer] == 'gin':
                conv = GraphConv(input_hidden, output_hidden)
                conv.reset_parameters()
            elif gnn_method[layer] == 'gat':
                conv = GATConv(in_channels=input_hidden, out_channels=output_hidden, heads=num_heads, concat=False)
                conv.reset_parameters()
            else:
                raise NotImplementedError

            self.gcnlayers.append(conv)

        if self.use_gnn_norm:
            self.norms = torch.nn.ModuleList()
            for layer in range(self.num_layers-1):
                self.norms.append(IBN(n_hidden, self.IN_Ratio))
        # else:
        #     self.norms = torch.nn.ModuleList()
        #     for layer in range(self.num_layers):
        #         self.norms.append(BatchNorm(n_hidden))
        self.drop_out_ratio = drop_out_ratio

    def forward(self, data):
        return self.GraphNetForward(data)

    def GraphNetForward(self, data):
        x = data.x
        edge_index = data.edge_index
        for i in range(self.num_layers - 2):
            if self.gnn_method[i] in ['gcn', 'leconv', 'graphconv']:
                x = self.gcnlayers[i](x, edge_index)
            else:
                x = self.gcnlayers[i](x, edge_index)

            if self.use_gnn_norm:
                x = self.norms[i](x)
            x = F.relu(x)

            # mid_node_feats.append(x)
        x = self.gcnlayers[self.num_layers-2](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_out_ratio, training=self.training)

        # gat
        x = self.gcnlayers[self.num_layers - 1](x, edge_index)
        if self.use_gnn_norm:
            x = self.norms[-1](x)
        x = F.relu(x)
        fea = x

        return fea

class GNNModel(nn.Module):
    def __init__(self, in_feats_intra=1024, n_hidden_intra=1024,
                 out_feats_intra=1024, drop_out_ratio=0.1, IN_Ratio=0.1,
                 gnn_intra=['sage'], num_heads=1, mpool_inter="mean", use_gnn_norm=False):
        super(GNNModel, self).__init__()

        self.mpool_inter = mpool_inter
        # # intra-graph
        self.gcn1 = Res_GCN(
            in_feats=in_feats_intra, n_hidden=n_hidden_intra,
            out_feats=out_feats_intra,
            gnn_method=gnn_intra,
            num_heads=num_heads,
            use_gnn_norm=use_gnn_norm,
            IN_Ratio=IN_Ratio,
            drop_out_ratio=drop_out_ratio
        )

        self.os_head = nn.Linear(out_feats_intra, 1, bias=False)
        self.timefit_head = nn.Linear(out_feats_intra, 1, bias=False)

    def forward(self, batch_pyg, domain_labels=None):
        n_feat = self.gcn1(batch_pyg)
        if self.mpool_inter == 'mean':
            g_feat = global_mean_pool(n_feat, batch_pyg.batch)
        # print(self.os_head(g_feat))
        os_time = self.os_head(g_feat)
        # g_id = self.gcls_head(g_feat)
        time_fit = self.timefit_head(g_feat)
        return {
            'g_feat': g_feat,
            'os_time': os_time,
            'time_fit': time_fit,
        }

if __name__ == '__main__':
    from torch_geometric.profile import count_parameters, get_model_size
    def params_to_string(params_num: int, units=None,
                         precision: int = 2) -> str:
        """
        Converts integer params representation to a readable string.

        :param flops: Input number of parameters.
        :param units: Units for string representation of params (M, K or B).
        :param precision: Floating point precision for representing params in
            given units.
        """
        if units is None:
            if params_num // 10 ** 6 > 0:
                return str(round(params_num / 10 ** 6, precision)) + ' M'
            elif params_num // 10 ** 3:
                return str(round(params_num / 10 ** 3, precision)) + ' k'
            else:
                return str(params_num)
        else:
            if units == 'M':
                return str(round(params_num / 10. ** 6, precision)) + ' ' + units
            elif units == 'K':
                return str(round(params_num / 10. ** 3, precision)) + ' ' + units
            elif units == 'B':
                return str(round(params_num / 10. ** 9, precision)) + ' ' + units
            else:
                return str(params_num)

    test_model = GNNModel(in_feats_intra=512,
             n_hidden_intra=512,
             out_feats_intra=1024,
             gnn_intra=['gin', 'gin', 'gin', 'gat'],
             mpool_inter='mean',
             use_gnn_norm=True
             )

    params_count = count_parameters(test_model)
    print(params_to_string(params_count))
