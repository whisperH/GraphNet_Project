import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# https://github.com/XingangPan/Switchable-Whitening/blob/master/models/utils/norm.py
class SwitchWhiten1d(Module):
    """Switchable Whitening.

    Args:
        num_features (int): Number of channels.
        num_pergroup (int): Number of channels for each whitening group.
        sw_type (int): Switchable whitening type, from {2, 3, 5}.
            sw_type = 2: BW + IW
            sw_type = 3: BW + IW + LN
            sw_type = 5: BW + IW + BN + IN + LN
        T (int): Number of iterations for iterative whitening.
        tie_weight (bool): Use the same importance weight for mean and
            covariance or not.
    """

    def __init__(self,
                 num_features,
                 num_pergroup=16,
                 sw_type=2,
                 T=5,
                 tie_weight=False,
                 eps=1e-5,
                 momentum=0.99,
                 affine=True):
        super(SwitchWhiten1d, self).__init__()
        if sw_type not in [2, 3, 5]:
            raise ValueError('sw_type should be in [2, 3, 5], '
                             'but got {}'.format(sw_type))
        assert num_features % num_pergroup == 0
        self.num_features = num_features
        self.num_pergroup = num_pergroup
        self.num_groups = num_features // num_pergroup
        self.sw_type = sw_type
        self.T = T
        self.tie_weight = tie_weight
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        num_components = sw_type

        self.sw_mean_weight = Parameter(torch.ones(num_components))
        if not self.tie_weight:
            self.sw_var_weight = Parameter(torch.ones(num_components))
        else:
            self.register_parameter('sw_var_weight', None)

        if self.affine:
            self.weight = Parameter(torch.ones(num_features))
            self.bias = Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean',
                             torch.zeros(self.num_groups, num_pergroup, 1))
        self.register_buffer(
            'running_cov',
            torch.eye(num_pergroup).unsqueeze(0).repeat(self.num_groups, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_cov.zero_()
        nn.init.ones_(self.sw_mean_weight)
        if not self.tie_weight:
            nn.init.ones_(self.sw_var_weight)
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def __repr__(self):
        return ('{name}({num_features}, num_pergroup={num_pergroup}, '
                'sw_type={sw_type}, T={T}, tie_weight={tie_weight}, '
                'eps={eps}, momentum={momentum}, affine={affine})'.format(
                    name=self.__class__.__name__, **self.__dict__))


    def forward(self, x):
        N, C = x.size()
        c, g = self.num_pergroup, self.num_groups

        in_data_t = x.transpose(0, 1).contiguous()
        # g x c x (N x H x W)
        in_data_t = in_data_t.view(g, c, -1) # 32, 16, BS

        # calculate batch mean and covariance
        if self.training:
            # g x c x 1
            mean_bn = in_data_t.mean(-1, keepdim=True) # g, c, 1
            in_data_bn = in_data_t - mean_bn
            # g x c x c
            cov_bn = torch.bmm(in_data_bn,
                               in_data_bn.transpose(1, 2)).div(N)

            self.running_mean.mul_(self.momentum)
            self.running_mean.add_((1 - self.momentum) * mean_bn.data)
            self.running_cov.mul_(self.momentum)
            self.running_cov.add_((1 - self.momentum) * cov_bn.data)
        else:
            mean_bn = torch.autograd.Variable(self.running_mean)
            cov_bn = torch.autograd.Variable(self.running_cov)

        mean_bn = mean_bn.view(1, g, c, 1).expand(N, g, c, 1).contiguous()
        mean_bn = mean_bn.view(N * g, c, 1)
        cov_bn = cov_bn.view(1, g, c, c).expand(N, g, c, c).contiguous()
        cov_bn = cov_bn.view(N * g, c, c)

        # (N x g) x c x (H x W)
        in_data = x.view(N * g, c, -1)

        eye = in_data.data.new().resize_(c, c)
        eye = torch.nn.init.eye_(eye).view(1, c, c).expand(N * g, c, c)

        # calculate other statistics
        # (N x g) x c x 1
        mean_in = in_data.mean(-1, keepdim=True)
        x_in = in_data - mean_in
        # (N x g) x c x c
        cov_in = torch.bmm(x_in, torch.transpose(x_in, 1, 2)).div(1)
        if self.sw_type in [3, 5]:
            x = x.view(N, -1)
            mean_ln = x.mean(-1, keepdim=True).view(N, 1, 1, 1)
            mean_ln = mean_ln.expand(N, g, 1, 1).contiguous().view(N * g, 1, 1)
            var_ln = x.var(-1, keepdim=True).view(N, 1, 1, 1)
            var_ln = var_ln.expand(N, g, 1, 1).contiguous().view(N * g, 1, 1)
            var_ln = var_ln * eye
        if self.sw_type == 5:
            var_bn = torch.diag_embed(torch.diagonal(cov_bn, dim1=-2, dim2=-1))
            var_in = torch.diag_embed(torch.diagonal(cov_in, dim1=-2, dim2=-1))

        # calculate weighted average of mean and covariance
        softmax = nn.Softmax(0)
        mean_weight = softmax(self.sw_mean_weight)
        if not self.tie_weight:
            var_weight = softmax(self.sw_var_weight)
        else:
            var_weight = mean_weight

        # BW + IW
        if self.sw_type == 2:
            # (N x g) x c x 1
            mean = mean_weight[0] * mean_bn + mean_weight[1] * mean_in
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                self.eps * eye
        # BW + IW + LN
        elif self.sw_type == 3:
            mean = mean_weight[0] * mean_bn + \
                mean_weight[1] * mean_in + mean_weight[2] * mean_ln
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                var_weight[2] * var_ln + self.eps * eye
        # BW + IW + BN + IN + LN
        elif self.sw_type == 5:
            mean = (mean_weight[0] + mean_weight[2]) * mean_bn + \
                (mean_weight[1] + mean_weight[3]) * mean_in + \
                mean_weight[4] * mean_ln
            cov = var_weight[0] * cov_bn + var_weight[1] * cov_in + \
                var_weight[0] * var_bn + var_weight[1] * var_in + \
                var_weight[4] * var_ln + self.eps * eye

        # perform whitening using Newton's iteration
        Ng, c, _ = cov.size()
        P = torch.eye(c).to(cov).expand(Ng, c, c)
        # reciprocal of trace of covariance
        rTr = (cov * P).sum((1, 2), keepdim=True).reciprocal_()
        cov_N = cov * rTr
        for k in range(self.T):
            P = torch.baddbmm(1.5, P, -0.5, torch.matrix_power(P, 3), cov_N)
        # whiten matrix: the matrix inverse of covariance, i.e., cov^{-1/2}
        wm = P.mul_(rTr.sqrt())

        x_hat = torch.bmm(wm, in_data - mean)
        x_hat = x_hat.view(N, C, 1, 1)
        if self.affine:
            x_hat = x_hat * self.weight.view(1, self.num_features, 1, 1) + \
                self.bias.view(1, self.num_features, 1, 1)

        return x_hat.squeeze()

# https://github.com/SY-Xuan/IIDS/blob/main/reid/models/backbones/AIBN.py#L5
class AIBNorm1d(nn.Module):
    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.9,
                 using_moving_average=True,
                 only_bn=False,
                 last_gamma=False,
                 adaptive_weight=None,
                 generate_weight=True,
                 init_weight=0.1):
        super(AIBNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.only_bn = only_bn
        self.last_gamma = last_gamma
        self.generate_weight = generate_weight

        if generate_weight:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        if not only_bn:
            if adaptive_weight is not None:
                self.adaptive_weight = adaptive_weight
            else:
                self.adaptive_weight = nn.Parameter(
                    torch.ones(1) * init_weight)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        self.reset_parameters()

    def reset_parameters(self):

        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x, weight=None, bias=None):
        N, C = x.size()
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        temp = var_in + mean_in**2

        if self.training:
            mean_bn = mean_in.mean(0, keepdim=True)
            var_bn = temp.mean(0, keepdim=True) - mean_bn**2
            if self.using_moving_average:
                self.running_mean.mul_(self.momentum)
                self.running_mean.add_(
                    (1 - self.momentum) * mean_bn.squeeze().data)
                self.running_var.mul_(self.momentum)
                self.running_var.add_(
                    (1 - self.momentum) * var_bn.squeeze().data)
            else:
                self.running_mean.add_(mean_bn.squeeze().data)
                self.running_var.add_(mean_bn.squeeze().data**2 +
                                      var_bn.squeeze().data)
        else:
            mean_bn = torch.autograd.Variable(
                self.running_mean).unsqueeze(0).unsqueeze(2)
            var_bn = torch.autograd.Variable(
                self.running_var).unsqueeze(0).unsqueeze(2)

        if not self.only_bn:

            adaptive_weight = torch.clamp(self.adaptive_weight, 0, 1)
            mean = (1 - adaptive_weight[0]
                    ) * mean_in + adaptive_weight[0] * mean_bn
            var = (1 -
                   adaptive_weight[0]) * var_in + adaptive_weight[0] * var_bn

            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(N, C)
        else:
            x = (x - mean_bn) / (var_bn + self.eps).sqrt()
            x = x.view(N, C)

        if self.generate_weight:
            weight = self.weight.view(1, self.num_features)
            bias = self.bias.view(1, self.num_features)
        else:
            weight = weight.view(1, self.num_features)
            bias = bias.view(1, self.num_features)
        return x * weight + bias

# https://github.com/XingangPan/IBN-Net/blob/master/ibnnet/modules.py#L5
class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm1d(self.half, affine=True)
        self.BN = nn.BatchNorm1d(planes - self.half)


    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].t().contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1.t(), out2), 1)
        return out

# https://github.com/lsj2408/GraphNorm/blob/master/GraphNorm_ws/gnn_ws/gnn_example/model/Norm/norm.py
class GraphNorm(nn.Module):
    """
        Param: []
    """
    def __init__(self, norm_type, hidden_dim=64, print_info=None):
        super(GraphNorm, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.print_info = print_info
        if norm_type == 'bn':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'gn':
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, graph, tensor, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = torch.unique(graph.batch,return_counts=True)[1]
        batch_size = graph.batch_size
        batch_index = graph.batch
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class GroupingUnit(nn.Module):
    def __init__(self, in_channels, num_parts):
        super(GroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels

        # params
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels))  # n * 1024 * 1*1
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            # msra init
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            # init smooth factor based on clustering
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, inputs):
        # inputs: Node size, feat_dim

        inputs = inputs.unsqueeze(2).unsqueeze(3)

        # 0. store input size
        node_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # 1. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        grouping_centers = self.weight.contiguous().view(1, self.num_parts, self.in_channels).expand(node_size,
                                                                                                     self.num_parts,
                                                                                                     self.in_channels)

        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(node_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(node_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(node_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(node_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(node_size, self.num_parts, -1) # assign.size() 320, 16, 1
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)

        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)

        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(
            node_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C
        outputs = nn.functional.normalize(out, dim=2)  # b 5 1024
        outputs_t = outputs.permute(0, 2, 1)  # b 1024 5

        # generate assignment map for basis for visualization
        # outputs_t: node_size, feat_dim, Group_num
        # assign: node_size, Group_num, 1, 1
        return outputs_t, assign.squeeze()

    # name
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.num_parts) + ')'


class GroupingUnit2D(nn.Module):
    def __init__(self, in_channels, num_parts, merge_method):
        super(GroupingUnit2D, self).__init__()
        self.num_parts = num_parts
        self.merge_method = merge_method
        self.in_channels = in_channels

        # params
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels))  # n * 1024 * 1*1
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            # msra init
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            # init smooth factor based on clustering
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, mid_node_feats, batch_list):
        # inputs: Node size, feat_dim

        inputs = torch.cat(mid_node_feats, dim=0)

        # inputs = nn.functional.normalize(inputs, dim=1).unsqueeze(2).unsqueeze(3)
        inputs = inputs.unsqueeze(2).unsqueeze(3)
        # 0. store input size
        node_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # 1. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        grouping_centers = self.weight.contiguous().view(1, self.num_parts, self.in_channels).expand(node_size,
                                                                                                     self.num_parts,
                                                                                                     self.in_channels)

        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(node_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(node_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(node_size, -1, input_h, input_w)
        sigma = (beta / 2).sqrt()
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # compute weighted feats N * K * C
        assign_list = assign.split(mid_node_feats[0].size(0))
        input_list = inputs.split(mid_node_feats[0].size(0))
        scale_feats = {}
        for scale_idx, (iscale_node_g_assigns, iscale_merge_node_g_feats) in enumerate(zip(assign_list, input_list)):
            node_g_assigns = iscale_node_g_assigns.squeeze().split(batch_list)
            merge_node_g_feats = iscale_merge_node_g_feats.squeeze().split(batch_list)
            for ig_ind, (ig_nodes_feat, ig_node_assign) in enumerate(zip(merge_node_g_feats, node_g_assigns)):
                if ig_ind not in scale_feats:
                    scale_feats[ig_ind] = {
                        "ig_node_feat": [],
                        "ig_node_assign": [],
                    }
                scale_feats[ig_ind]['ig_node_feat'].append(ig_nodes_feat)
                scale_feats[ig_ind]['ig_node_assign'].append(ig_node_assign)

        grouping_feats = []
        for ig_idx, g_info in scale_feats.items():
            ig_feats = torch.cat(g_info['ig_node_feat'], dim=0)
            ig_assign = torch.cat(g_info['ig_node_assign'], dim=0)
            if self.merge_method == "weighted_sum":
                merge_node_feat = ig_assign.t().matmul(ig_feats)
                sum_ass = ig_assign.sum(0).unsqueeze(1)

                # merge_node_feat = ((merge_node_feat / sum_ass) - grouping_centers[0]) / sigma.unsqueeze(1)
                merge_node_feat = ig_assign.t().matmul(ig_feats)
                # if not self.training:
                #     print(torch.topk(ig_assign, 1)[1].t())
            elif self.merge_method == "assign_avg":
                merge_node_feat = torch.zeros((self.num_parts, self.in_channels)).to(ig_feats.device)
            grouping_feats.append(merge_node_feat.unsqueeze(0))
            outputs = torch.cat(grouping_feats, dim=0)


        return outputs, assign.squeeze()

    # name
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.num_parts) + ')'