import os
from idlelib.iomenu import encoding

import matplotlib
matplotlib.use('TkAgg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import nmslib
import numpy as np
from torch_geometric.data import Data as geomData
from itertools import chain
import torch
import json
from bisect import bisect_left
# from utils.block_utils import visualize_tsne
from utils.block_utils import get_split_list


def max_min_transform(OS_time, max_live=100, min_live=0):
    max_os_time = float(max_live)
    min_os_time = float(min_live)
    scale_os_time = (OS_time - min_os_time) / (max_os_time - min_os_time)
    return scale_os_time

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def scaled_sigmoid(x, b=90):
    """
    when x in start and end, the value should be in [0, 1]
    """

    score = 2/(1+np.exp(-np.log(40000)*(x-b)/b+np.log(5e-3)))
    return score / 2

class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices, dist

def generate_tissue_graph(coords, features, patch_conf, p_id, OS_time, events, Domain_id, split_list, radius=9):
    # https://github.com/pyg-team/pytorch_geometric/discussions/3442
    model = Hnsw(space='l2')
    model.fit(coords)

    num_patches = coords.shape[0]
    # np.repeat: https://blog.csdn.net/zyl1042635242/article/details/43052403
    if num_patches <= radius - 1:
        small_radius = num_patches // 2
        a = np.repeat(range(num_patches), small_radius)
        qinds, qdists = [], []
        for v_idx in range(num_patches):
            qind, qdist = model.query(coords[v_idx], topn=small_radius+1)
            qinds.append(qind[1:])
            qdists.append(qdist[1:])
        b = np.fromiter(chain(*qinds), dtype=int)
        # c = np.fromiter(chain(*qdists), dtype=int)
    else:
        a = np.repeat(range(num_patches), radius - 1)
        qinds, qdists = [], []
        for v_idx in range(num_patches):
            qind, qdist = model.query(coords[v_idx], topn=radius)
            qinds.append(qind[1:])
            qdists.append(qdist[1:])
        b = np.fromiter(chain(*qinds), dtype=int)
        #
    c = 1.5 - normalization(qdists)
    edge_dist = np.fromiter(chain(*c), dtype=float)
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)
    edge_dist = torch.Tensor(edge_dist).type(torch.FloatTensor)

    scale_OS_time = max_min_transform(OS_time)

    c_id = bisect_left(split_list, scale_OS_time) - 1
    if events == 1: # 事件发生
        pseudo_id = c_id + len(split_list)
    elif events == 0:
        pseudo_id = c_id
    elif events == -1:
        pseudo_id = -1
    else:
        print(events)
        raise "unknown events {}".format(events)

    G = geomData(
        edge_index=edge_spatial,
        edge_weight=edge_dist.unsqueeze(1),
        # centroid=torch.Tensor(coords),
        y=torch.Tensor([OS_time]).unsqueeze(0),
        y_day=torch.Tensor([OS_time]).unsqueeze(0),
        domain_id=torch.Tensor([Domain_id]).unsqueeze(0),
        graph_id = torch.Tensor([p_id]).unsqueeze(0).type(torch.LongTensor),
        cluster_id = torch.Tensor([pseudo_id]).unsqueeze(0).type(torch.LongTensor),
        events = torch.Tensor([events]).unsqueeze(0),
        x=torch.Tensor(features).squeeze(),
        # node_conf=torch.Tensor([patch_conf])
    )
    return G


def get_instance_info(instance_path, p_idx, OS_time, events, Domain_id, split_list, use_patch_nums):
    coords = []
    feats = []
    patch_confs = []

    feat_files = sorted([_ for _ in os.listdir(instance_path) if _.endswith(".npz") and "_norm" not in _])
    img_files = sorted([_ for _ in os.listdir(instance_path) if _.endswith(".png")])
    assert len(feat_files) == len(img_files), "unmatching with feature and images"
    if len(feat_files) <= 1:
        print(f"{instance_path} with no patches or with 1 patch")
        exit(-1)
    else:
        for patch_id, (ipatch_feat_file, ipatch_img_file) in enumerate(zip(feat_files, img_files)):
            file_name = ipatch_feat_file.split(".npz")[0]
            img_name = ipatch_img_file.split(".png")[0]
            assert img_name == file_name, "unmatch with filename and img_name"
            patch_feat = np.load(os.path.join(instance_path, ipatch_feat_file))['emb']
            *raw_name, coordx, coordy, conf = file_name.split("_")
            # print(raw_name)
            # print([coordx, coordy])
            # print(float(conf))
            coords.append([int(coordx), int(coordy)])
            feats.append(patch_feat)
            patch_confs.append(float(conf))

            # ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if patch_id == use_patch_nums:
                break
    return generate_tissue_graph(
        np.array(coords), np.array(feats), np.array(patch_confs),
        p_idx, OS_time, events, Domain_id, split_list
    )



def main(args):

    graph_file_saved_path = args.graph_file_saved_path+"".join(args.norm_stain)

    os.makedirs(graph_file_saved_path, exist_ok=True)

    domain_pid = 0

    split_list = get_split_list(args.split_num, args.ultra_split_num, args.ultra_start, args.ultra_end)

    all_feats = []
    all_domain_labels = []
    all_pid_labels = []


    for domain_id, json_file in json_files.items():
        all_Gid_PID_maps = {}
        domain_name = json_file.split(".json")[0]
        print(f"Processing json data {domain_name}")
        with open(os.path.join(args.json_dir, json_file), 'r', encoding='utf-8') as load_f:
            load_values = json.load(load_f)

        # all_samples = []
        # for isample in load_values:
        #     if json_file == "rfs_new_cyhz.json":
        #         all_samples.append(isample["PID"].split(" ")[0])
        #     else:
        #         all_samples.append(isample["name"])
        # all_set_name = len(set(all_samples))
        print(f"length of json data {len(load_values)}")
        graphs = []
        domain_feats = []
        for p_idx, isample in enumerate(load_values):
            # if args.vis_tsne:
            #     if p_idx > 75:
            #         print(f"jump {p_idx}")
            #         break
            all_Gid_PID_maps[p_idx] = {
                'domain_id': domain_id,
                'PID': isample["PID"],
                'name': isample["name"],

            }
            instance_path = isample["path"].replace("/workspace/data1/medicine", "/home/whisper/Disk2/data")
            OS_time = isample["RFS_time"]
            events = isample["events"]

            win_instance_path = os.path.join(args.parent_dir, instance_path)
            for inorm_stain in args.norm_stain:
                if inorm_stain == "None":
                    img_dir = win_instance_path
                else:
                    img_dir = win_instance_path + f"={inorm_stain}"
                # print(f"processing {img_dir}")
                if args.vis_tsne:
                    exit(-1)
                    feat_files = [_ for _ in os.listdir(img_dir) if _.endswith(".npz")]
                    for ipatch_feat_file in feat_files:
                        patch_feat = np.load(os.path.join(img_dir, ipatch_feat_file))['emb']
                        domain_feats.append(patch_feat)
                        all_feats.append(patch_feat)
                        all_domain_labels.append(domain_id)
                        all_pid_labels.append(p_idx)
                else:
                    instance_info = get_instance_info(
                        img_dir, p_idx=p_idx, OS_time=OS_time,
                        events=events, Domain_id=domain_id, split_list=split_list, use_patch_nums=args.use_patch_nums
                    )
                    graphs.append(instance_info)
                    # if "Toy" in graph_file_saved_path:
                        # if p_idx > 200 and p_idx < 350:
                        #     print(f"{p_idx}Toy Down")
                        #     break
        # print(f"{len(domain_feats)} feats in {json_file}")
        print(f"{len(graphs)} graphs in {json_file}")
        domain_pid = domain_pid+1
        torch.save(graphs, os.path.join(graph_file_saved_path, f'{domain_name}.pt'))

        print(f"length of graph {len(all_Gid_PID_maps)}")
        assert len(all_Gid_PID_maps) == len(load_values), 'unmatch length of graphs and json values'
        with open(os.path.join(graph_file_saved_path, f'{domain_name}_GP_maps.json'), 'w+', encoding='utf-8') as f:
            json.dump(all_Gid_PID_maps, f, ensure_ascii=False)

    if args.vis_tsne:
        exit(-1)
        # visualize_tsne(np.array(all_feats).squeeze(), all_domain_labels, json_files)


def get_params():
    parser = argparse.ArgumentParser(description='superpixel_generate')

    parser.add_argument('--parent_dir', type=str, default='../dataset/Patch_Images')
    parser.add_argument('--json_dir', type=str, default='../dataset/RFS_Data_UpLoad')
    parser.add_argument('--graph_file_saved_path', type=str, default='../dataset/GraphData')
    # parser.add_argument('--norm_stain', nargs='+', default=["None", 'Vahadane', "Reinhard", "Ruifrok", "Macenko"]) # None, Reinhard, Ruifrok Macenko Vahadane
    parser.add_argument('--norm_stain', nargs='+', default=["None"]) # None, Reinhard, Ruifrok Macenko Vahadane
    parser.add_argument('--max_live', type=float, default=150)
    parser.add_argument('--min_live', type=float, default=0)
    parser.add_argument('--scale_live', type=float, default=90)
    parser.add_argument('--split_num', type=float, default=500)
    parser.add_argument('--ultra_split_num', type=float, default=0)
    parser.add_argument('--ultra_start', type=float, default=9)
    parser.add_argument('--ultra_end', type=float, default=50)
    parser.add_argument('--use_patch_nums', type=int, default=20)
    parser.add_argument('--vis_tsne',  default=False)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    json_files = {
        0: 'rfs_CY.json',
        1: 'rfs_huashan.json',
        2: 'rfs_huashan2.json',
        3: 'rfs_hz.json',
        4: 'rfs_YouAn.json',
        5: 'rfs_JiangData32.json',
    }

    try:
        args=get_params()
        main(args)

    except Exception as exception:
#         logger.exception(exception)
        raise