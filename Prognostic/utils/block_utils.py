import torch
import torch.nn as nn
# from timm.models.layers import Mlp,DropPath
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from heapq import nsmallest

from collections import Counter


def analysis_sampler(pyg_data, sampler_list):
    domain_list = []
    rfs_time = []
    for epoch, sub_sample_list in enumerate(sampler_list):
        _domain_list = []
        _rfs_time = []
        for idx in sub_sample_list:
            idata = pyg_data[idx]
            _domain_list.append(int(idata.domain_id))
            _rfs_time.append(float(idata.y))

        domain_list.extend(_domain_list)
        rfs_time.extend(_rfs_time)

        result = Counter(_domain_list)
        print(f"for epoch {epoch} Domain Count:")
        print(result)
        # result = Counter(rfs_time)
        # print(f"epoch {epoch} rfs_time Count:")
        # print(result)
    result = Counter(domain_list)
    print("Domain Count:")
    print(result)
    # result = Counter(rfs_time)
    # print("rfs_time Count:")
    # print(result)

import os
def setup_seed(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.weight, 0.005)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.attention_weights: Optional[Tensor] = None
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         # print(attn.shape)
#         self.attention_weights = attn
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#     def get_attention_weights(self):
#         return self.attention_weights


# class LayerScale(nn.Module):
#     def __init__(self, dim, init_values=1e-5, inplace=False):
#         super().__init__()
#         self.inplace = inplace
#         self.gamma = nn.Parameter(init_values * torch.ones(dim))
#
#     def forward(self, x):
#         return x.mul_(self.gamma) if self.inplace else x * self.gamma




# class Block(nn.Module):
#
#     def __init__(
#             self,
#             dim,
#             num_heads,
#             mlp_ratio=4.,
#             qkv_bias=False,
#             drop=0.,
#             attn_drop=0.,
#             init_values=None,
#             drop_path=0.,
#             act_layer=nn.GELU,
#             norm_layer=nn.LayerNorm
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
#         self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
#         self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x):
#         x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x
#
#     def get_attention_weights(self):
#         return self.attn.get_attention_weights()


def correct_stat(pred, y_label):
    prob = pred.argmax(dim=1)  # Use the class with highest probability.
    correct = int((prob == y_label).sum())  # Check
    return correct / y_label.shape[0] * 100

def visualize_tsne(feats, targets, label_names):
    t_sne_features = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=2022).fit_transform(feats)
    np_targets = np.array(targets)
    color = ['black', 'blue', 'olive', "darkred", "orchid", "pink"]
    marker = ['x', 'o', ">", "D", "*"]
    for i in set(targets):
        print(i)
        iidx = np_targets==i
        plt.scatter(
            x=t_sne_features[iidx, 0],
            y=t_sne_features[iidx, 1],
            # c=color[i],
            label=label_names[i], cmap='jet', s=10
            # , marker=marker[i]
        )
    plt.legend()
    plt.show()

def select_cluster(domain_cids, used_cid, cid_per_batch):
    unuse_cid = np.setdiff1d(
        # data_info['domain2cid'][idomain],
        domain_cids,
        used_cid
    )
    if len(unuse_cid) < cid_per_batch:
        cid_select1 = np.random.choice(
            # data_info['domain2cid'][idomain],
            domain_cids,
            cid_per_batch - len(unuse_cid),
            replace=False
        )
        cid_select = np.hstack((unuse_cid, cid_select1))
    else:
        cid_select = np.random.choice(unuse_cid, cid_per_batch, replace=False)
    assert len(cid_select) == cid_per_batch, "unmatch with cid per batch"
    return cid_select

def select_neighbor_cluster(centroid_cid, Edomain_cids, Eused_cid, cid_per_batch, win_size):
    # 两种可能：
    # 同一 事件中相邻的值；不同事件中相邻的值，对应的class id相差split_num，参见generate_instance_graph: split_num
    Eunuse_cid = np.setdiff1d(
        # data_info['domain2cid'][idomain],
        Edomain_cids,
        Eused_cid
    )
    if len(Eunuse_cid) < cid_per_batch:
        Neighbors_cid = nsmallest(win_size, Edomain_cids, key=lambda x: abs(x - centroid_cid))
        cid_select1 = np.random.choice(
            # data_info['domain2cid'][idomain],
            Neighbors_cid,
            cid_per_batch - len(Eunuse_cid),
            replace=False
        )
        cid_select = np.hstack((Eunuse_cid, cid_select1))
    else:
        Neighbors_cid = nsmallest(win_size, Eunuse_cid, key=lambda x: abs(x - centroid_cid))
        cid_select = np.random.choice(Neighbors_cid, cid_per_batch, replace=False)
    assert len(cid_select) == cid_per_batch, "unmatch with cid per batch"
    return cid_select

def select_instance(domain_cids, cid_select, used_data_index, used_cid, instance_per_cid):
    sampler_sublist = []
    for icid_select in cid_select:
        # 剩下的 cid_data_index
        un_usedata_index = np.setdiff1d(
            domain_cids[icid_select],
            used_data_index
        )
        cid_data_index_len = len(un_usedata_index)
        if cid_data_index_len <= instance_per_cid:
            # 如果剩余列表中的instance不够了，先把多余的加到list中
            sampler_sublist.extend(un_usedata_index)
            # 如果原本列表数量大于 instance_per_cid ，从原列表中随机选择 instance_per_cid-cid_data_index_len个
            if len(domain_cids[icid_select]) > instance_per_cid:
                sampler_sublist.extend(
                    np.random.choice(
                        domain_cids[icid_select],
                        instance_per_cid - cid_data_index_len,
                        replace=False).tolist()
                )
            else:
                # 如果原本列表数量小于 instance_per_cid ，先重复再选择
                repeat_num = instance_per_cid // len(domain_cids[icid_select])
                tmp_list = np.repeat(domain_cids[icid_select], repeat_num + 1)
                sampler_sublist.extend(
                    np.random.choice(
                        tmp_list,
                        instance_per_cid - cid_data_index_len,
                        replace=False
                    ).tolist()
                )
            used_cid.append(icid_select)
        else:
            sampler_sublist.extend(
                np.random.choice(
                    domain_cids[icid_select],
                    instance_per_cid,
                    replace=False)
            )
        used_data_index.extend(sampler_sublist)
    return sampler_sublist, used_cid, list(set(used_data_index))


def generate_cid_label(cid_label, Event0_cid_select, Event1_cid_select, instance_per_cid):
    classify_label1 = [
        cid_label[_] for _ in np.repeat(Event0_cid_select, instance_per_cid)
    ]
    classify_label2 = [
        cid_label[_] for _ in np.repeat(Event1_cid_select, instance_per_cid)
    ]
    return classify_label1+classify_label2

def get_split_list(split_num, ultra_split_num=2100, ultra_start=9, ultra_end=50):
    if ultra_split_num > 0:
        split_area0 = [_ / float(split_num) for _ in range(int(split_num))]
        split_area1 = [_ / ultra_split_num for _ in range(ultra_start, ultra_end)]
        split_list = np.sort(split_area0 + split_area1)
        return split_list
    else:
        split_list = [_ / float(split_num) for _ in range(int(split_num))]
        return split_list

def check_all_same(arr):
    return all(x == arr[0] for x in arr)