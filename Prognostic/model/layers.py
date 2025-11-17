import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import warnings
from typing import Optional, Tuple, Union, List

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
    def __init__(self, planes, ratio=0.1):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm1d(self.half, affine=True)
        self.BN = nn.BatchNorm1d(planes - self.half)


    def forward(self, x):
        # split = torch.split(x, self.half, 1)
        # out1 = self.IN(split[0].t().contiguous())
        # out2 = self.BN(torch.cat(split[1:], dim=1).contiguous())
        out1 = self.IN(x[:, :self.half].t().contiguous())
        out2 = self.BN(x[:, self.half:].contiguous())
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



class MLP(torch.nn.Sequential):
    """MLP Module.

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    hidden: Optional[List[int]] = None
        Dimension of hidden layer(s).
    dropout: Optional[List[float]] = None
        Dropout rate(s).
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        MLP activation.
    bias: bool = True
        Add bias to MLP hidden layers.

    Raises
    ------
    ValueError
        If ``hidden`` and ``dropout`` do not share the same length.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        if dropout is not None:
            if hidden is not None:
                assert len(hidden) == len(
                    dropout
                ), "hidden and dropout must have the same length"
            else:
                raise ValueError(
                    "hidden must have a value and have the same length as dropout if dropout is given."
                )

        d_model = in_features
        layers = []

        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [torch.nn.Linear(d_model, h, bias=bias)]
                d_model = h

                if activation is not None:
                    seq.append(activation)

                if dropout is not None:
                    seq.append(torch.nn.Dropout(dropout[i]))

                layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)

class MaskedLinear(torch.nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.
    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.mask_value = mask_value

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):  # pylint: disable=arguments-renamed
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, SEQ_LEN, IN_FEATURES).
        mask: Optional[torch.BoolTensor] = None
            True for values that were padded, shape (B, SEQ_LEN, 1),

        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mask_value={self.mask_value}, bias={self.bias is not None}"
        )


class TilesMLP(torch.nn.Module):
    """MLP to be applied to tiles to compute scores.
    This module can be used in combination of a mask
    to prevent padding from influencing the scores values.
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    hidden: Optional[List[int]] = None
        Number of hidden layers and their respective number of features.
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    activation: torch.nn.Module = torch.nn.Sigmoid()
        MLP activation function
    dropout: Optional[torch.nn.Module] = None
        Optional dropout module. Will be interlaced with the linear layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden: Optional[List[int]] = None,
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
        dropout: Optional[torch.nn.Module] = None,
    ):
        super(TilesMLP, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        if hidden is not None:
            for h in hidden:
                self.hidden_layers.append(
                    MaskedLinear(in_features, h, bias=bias, mask_value="-inf")
                )
                self.hidden_layers.append(activation)
                if dropout:
                    self.hidden_layers.append(dropout)
                in_features = h

        self.hidden_layers.append(
            torch.nn.Linear(in_features, out_features, bias=bias)
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):
        """Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES), True for values that were padded.

        Returns
        -------
        x: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x

class ExtremeLayer(torch.nn.Module):
    """Extreme layer.
    Returns concatenation of n_top top tiles and n_bottom bottom tiles
    .. warning::
        If top tiles or bottom tiles is superior to the true number of
        tiles in the input then padded tiles will be selected and their value
        will be 0.
    Parameters
    ----------
    n_top: Optional[int] = None
        Number of top tiles to select
    n_bottom: Optional[int] = None
        Number of bottom tiles to select
    dim: int = 1
        Dimension to select top/bottom tiles from
    return_indices: bool = False
        Whether to return the indices of the extreme tiles

    Raises
    ------
    ValueError
        If ``n_top`` and ``n_bottom`` are set to ``None`` or both are 0.
    """

    def __init__(
        self,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        dim: int = 1,
        return_indices: bool = False,
    ):
        super(ExtremeLayer, self).__init__()

        if not (n_top is not None or n_bottom is not None):
            raise ValueError("one of n_top or n_bottom must have a value.")

        if not (
            (n_top is not None and n_top > 0)
            or (n_bottom is not None and n_bottom > 0)
        ):
            raise ValueError("one of n_top or n_bottom must have a value > 0.")

        self.n_top = n_top
        self.n_bottom = n_bottom
        self.dim = dim
        self.return_indices = return_indices

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, N_TILES, IN_FEATURES).
        mask: Optional[torch.BoolTensor]
            True for values that were padded, shape (B, N_TILES, 1).

        Warnings
        --------
        If top tiles or bottom tiles is superior to the true number of tiles in
        the input then padded tiles will be selected and their value will be 0.

        Returns
        -------
        values: torch.Tensor
            Extreme tiles, shape (B, N_TOP + N_BOTTOM).
        indices: torch.Tensor
            If ``self.return_indices=True``, return extreme tiles' indices.
        """

        if (
            self.n_top
            and self.n_bottom
            and ((self.n_top + self.n_bottom) > x.shape[self.dim])
        ):
            warnings.warn(
                f"Sum of tops is larger than the input tensor shape for dimension {self.dim}: "
                f"{self.n_top + self.n_bottom} > {x.shape[self.dim]}. "
                f"Values will appear twice (in top and in bottom)"
            )

        top, bottom = None, None
        top_idx, bottom_idx = None, None
        if mask is not None:
            if self.n_top:
                top, top_idx = x.masked_fill(mask, float("-inf")).topk(
                    k=self.n_top, sorted=True, dim=self.dim
                )
                top_mask = top.eq(float("-inf"))
                if top_mask.any():
                    warnings.warn(
                        "The top tiles contain masked values, they will be set to zero."
                    )
                    top[top_mask] = 0

            if self.n_bottom:
                bottom, bottom_idx = x.masked_fill(mask, float("inf")).topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )
                bottom_mask = bottom.eq(float("inf"))
                if bottom_mask.any():
                    warnings.warn(
                        "The bottom tiles contain masked values, they will be set to zero."
                    )
                    bottom[bottom_mask] = 0
        else:
            if self.n_top:
                top, top_idx = x.topk(k=self.n_top, sorted=True, dim=self.dim)
            if self.n_bottom:
                bottom, bottom_idx = x.topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )

        if top is not None and bottom is not None:
            values = torch.cat([top, bottom], dim=self.dim)
            indices = torch.cat([top_idx, bottom_idx], dim=self.dim)
        elif top is not None:
            values = top
            indices = top_idx
        elif bottom is not None:
            values = bottom
            indices = bottom_idx
        else:
            raise ValueError

        if self.return_indices:
            return values, indices
        else:
            return values

    def extra_repr(self) -> str:
        """Format representation."""
        return f"n_top={self.n_top}, n_bottom={self.n_bottom}"