import os
from idlelib.iomenu import encoding

import matplotlib
from sympy import print_glsl

matplotlib.use('TkAgg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import nmslib
import random
import numpy as np
from torch_geometric.data import Data as geomData
from itertools import chain
import torch
import json

random.seed(42)


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

def generate_tissue_graph(coords, features, patch_conf, p_id, OS_time, events, Domain_id, split_list, link_edge, radius=9):
    num_patches = coords.shape[0]
    if link_edge == 'near8':
        # https://github.com/pyg-team/pytorch_geometric/discussions/3442
        model = Hnsw(space='l2')
        model.fit(coords)

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
        else:
            a = np.repeat(range(num_patches), radius - 1)
            qinds, qdists = [], []
            for v_idx in range(num_patches):
                qind, qdist = model.query(coords[v_idx], topn=radius)
                qinds.append(qind[1:])
                qdists.append(qdist[1:])
            b = np.fromiter(chain(*qinds), dtype=int)
        # c = 1.5 - normalization(qdists)

    elif link_edge == 'random':
        # random.seed(123)
        if num_patches <= radius - 1:
            small_radius = num_patches // 2
        else:
            small_radius = radius - 1
        a = np.repeat(range(num_patches), small_radius)
        b = []
        for q_idx in range(num_patches):
            b.extend(random.sample([i for i in range(max(a)) if i != q_idx], small_radius))
        # c = np.array([[-1]*small_radius] * num_patches)
    else:
        print("Unknown link edge type")
        exit(-1)

    # edge_dist = np.fromiter(chain(*c), dtype=float)
    edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)
    # edge_dist = torch.Tensor(edge_dist).type(torch.FloatTensor)
    # scale_OS_time = max_min_transform(OS_time)

    # c_id = bisect_left(split_list, scale_OS_time) - 1
    # if events == 1: # 事件发生
    #     pseudo_id = c_id + len(split_list)
    # elif events == 0:
    #     pseudo_id = c_id
    # elif events == -1:
    #     pseudo_id = -1
    # else:
    #     print(events)
    #     raise "unknown events {}".format(events)

    G = geomData(
        edge_index=edge_spatial,
        # edge_weight=edge_dist.unsqueeze(1),
        # centroid=torch.Tensor(coords),
        y=torch.Tensor([OS_time]).unsqueeze(0),
        # y_day=torch.Tensor([OS_time]).unsqueeze(0),
        domain_id=torch.Tensor([Domain_id]).unsqueeze(0),
        graph_id = torch.Tensor([p_id]).unsqueeze(0).type(torch.LongTensor),
        # cluster_id = torch.Tensor([pseudo_id]).unsqueeze(0).type(torch.LongTensor),
        events = torch.Tensor([events]).unsqueeze(0),
        x=torch.Tensor(features).squeeze(),
        # node_conf=torch.Tensor([patch_conf])
    )
    return G


def get_instance_info(instance_path, img_dir, p_idx, OS_time, events, Domain_id, split_list, use_patch_nums, link_edge):
    coords = []
    feats = []
    patch_confs = []

    if (("rfs_JiangData32" in instance_path) or
            ("rfs_file_202051101" in instance_path) or
            ("rfs_ZY4HE26605ZHL" in instance_path) or
            ("CY" in instance_path)
    ):
        all_feat_files = sorted([_ for _ in os.listdir(instance_path) if _.endswith(".npz") and "_norm" not in _])
        all_img_files = sorted([_ for _ in os.listdir(img_dir) if _.endswith(".png")])
        feat_files = sorted(all_feat_files, key=lambda x: float(x.split('_')[-1].replace('.npz', '')), reverse=True)
        img_files = sorted(all_img_files, key=lambda x: float(x.split('_')[-1].replace('.png', '')), reverse=True)
    else:
        feat_files = sorted([_ for _ in os.listdir(instance_path) if _.endswith(".npz") and "_norm" not in _])
        img_files = sorted([_ for _ in os.listdir(img_dir) if _.endswith(".png")])
    assert len(feat_files) == len(img_files), f"unmatch with features and images: {instance_path}"
    if len(feat_files) <= 1:
        print(f"{instance_path} with no patches or with 1 patch")
        exit(999)
    else:
        for patch_id, (ipatch_feat_file, ipatch_img_file) in enumerate(zip(feat_files, img_files)):
            file_name = ipatch_feat_file.split(".npz")[0]
            img_name = ipatch_img_file.split(".png")[0]
            assert img_name == file_name, f"unmatch with {file_name} and {img_name}"
            patch_feat = np.load(os.path.join(instance_path, ipatch_feat_file))['emb']
            *raw_name, coordx, coordy, conf = file_name.split("_")
            # print(raw_name)
            # print([coordx, coordy])
            # print(float(conf))
            coords.append([int(coordx), int(coordy)])
            feats.append(patch_feat)
            patch_confs.append(float(conf))

            # ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if patch_id >= use_patch_nums-1:
                break
        # if patch_id < use_patch_nums -1 :
        #     print(f"{instance_path} is {patch_id+1}")
        #     # exit(3333)

    assert len(coords) == 20
    return generate_tissue_graph(
        np.array(coords), np.array(feats), np.array(patch_confs),
        p_idx, OS_time, events, Domain_id, split_list, link_edge
    )



def main(args):

    graph_file_saved_path = os.path.join(args.parent_dir, args.graph_file_saved_path)

    os.makedirs(graph_file_saved_path, exist_ok=True)

    domain_pid = 0

    json_files = {
        0: f'rfs_{args.dataset_name}.json'
    }
    for domain_id, json_file in json_files.items():
        all_Gid_PID_maps = {}
        domain_name = json_file.split(".json")[0]
        print(f"Processing json data {domain_name}")
        # if args.json_dir is None:
        #     load_values = []
        #     embedding_filelist_dir = os.path.join(args.parent_dir, "patchEmbedding", args.dataset_name)
        #     for isample_name in os.listdir(embedding_filelist_dir):
        #         load_values.append({
        #             "RFS_daytime": -1,
        #             "RFS_time": -1,
        #             "events": -1,
        #             "PID": isample_name,
        #             "path": f"{args.dataset_name}/{isample_name}",
        #             "name": isample_name
        #         })
        # else:
        with open(os.path.join(args.json_dir, json_file), 'r', encoding='utf-8') as load_f:
            load_values = json.load(load_f)

        print(f"length of json data {len(load_values)}")

        graphs = []
        # domain_feats = []
        for p_idx, isample in enumerate(load_values):
            all_Gid_PID_maps[p_idx] = {
                'domain_id': domain_id,
                'PID': isample["PID"],
                'name': isample["name"],
                "events": isample["events"]
            }
            instance_path = isample["path"].replace("/workspace/data1/medicine", "/home/whisper/Disk2/data")
            OS_time = isample["RFS_time"]
            events = isample["events"]

            win_instance_path = os.path.join(args.parent_dir, "patchEmbedding", instance_path)
            for inorm_stain in args.norm_stain:
                if inorm_stain == "None":
                    embed_dir = win_instance_path
                else:
                    embed_dir = win_instance_path + f"={inorm_stain}"
                img_dir = embed_dir.replace("patchEmbedding", "patchImage")
                print(img_dir)
                print(embed_dir)
                instance_info = get_instance_info(
                    embed_dir, img_dir, p_idx=p_idx, OS_time=OS_time,
                    events=events, Domain_id=domain_id, split_list=None,
                    use_patch_nums=args.use_patch_nums,
                    link_edge=args.link
                )
                graphs.append(instance_info)
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
    # parser.add_argument('--json_dir', type=str, default=None)
    parser.add_argument('--json_dir', type=str, default='../dataset/RFS_Data_251026')
    # parser.add_argument('--dataset_name', type=str, default='CY')
    parser.add_argument('--dataset_name', type=str, default='CY')
    parser.add_argument('--graph_file_saved_path', type=str, default='../dataset/GraphDataTest') # GraphData_random_link, GraphData
    # parser.add_argument('--norm_stain', nargs='+', default=["None", 'Vahadane', "Reinhard", "Ruifrok", "Macenko"]) # None, Reinhard, Ruifrok Macenko Vahadane
    parser.add_argument('--norm_stain', nargs='+', default=["None"]) # None, Reinhard, Ruifrok Macenko Vahadane
    parser.add_argument('--link', type=str, default='near8') # random, near8
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
    try:
        # json_files = {
        #     0: 'rfs_CY.json',
        #     1: 'rfs_huashan.json',
        #     2: 'rfs_huashan2.json',
        #     3: 'rfs_hz.json',
        #     4: 'rfs_YouAn.json',
        #     5: 'rfs_JiangData32.json',
        #     6: 'rfs_file_202051101.json',
        #     7: 'rfs_ZY4HE26605ZHL.json',
        # }
        args=get_params()
        main(args)

    except Exception as exception:
#         logger.exception(exception)
        raise