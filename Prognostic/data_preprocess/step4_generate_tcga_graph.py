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
import h5py
import pandas as pd
import torch


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range



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
        random.seed(42)

        if num_patches <= radius - 1:
            small_radius = num_patches // 2 - 1
        else:
            small_radius = radius - 1
        a = np.repeat(range(num_patches), small_radius)
        b = []
        for q_idx in range(num_patches):
            b.extend(random.sample([i for i in range(max(a))], small_radius))
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


def get_instance_info(embedding_path, coord_path, p_idx, OS_time, events, Domain_id, split_list, use_patch_nums, link_edge):
    patch_confs = []

    feats = torch.load(embedding_path).numpy()
    wsi_h5 = h5py.File(coord_path, "r")
    coords = wsi_h5['coords'][:]
    return generate_tissue_graph(
        coords, feats, np.array(patch_confs),
        p_idx, OS_time, events, Domain_id, split_list, link_edge
    )



def main(args):

    graph_file_saved_path = args.graph_file_saved_path+"".join(args.norm_stain)

    os.makedirs(graph_file_saved_path, exist_ok=True)

    for domain_id, domain_name in tcga_data.items():
        all_Gid_PID_maps = {}

        print(f"Processing json data {domain_name}")
        tcga_anno_data = pd.read_csv(os.path.join(args.json_dir, f"tcga_{domain_name}_path_full.csv"))
        load_values = tcga_anno_data.T.to_dict()

        for ifold in range(5):
            tcga_5fold_split = pd.read_csv(
                os.path.join(args.json_dir, domain_name, f"{domain_name}_5fold/splits_{ifold}.csv")
            )
            all_Gid_PID_maps[ifold] = {
                'train': tcga_5fold_split['train'].tolist(),
                'test': tcga_5fold_split['val'].dropna(inplace=False).tolist(),
            }

        print(f"length of json data {len(load_values)}")
        train_graphs = {f"split_{id}": [] for id in range(5)}
        test_graphs = {f"split_{id}": [] for id in range(5)}
        # domain_feats = []
        for p_idx, isample in enumerate(load_values.items()):

            OS_time = isample[1]["t"] / 30.
            events = isample[1]["e"]
            embedding_path = os.path.join(args.parent_dir, domain_name, f"PatchEmbedding/{isample[1]['pathology_id']}.pt")
            coord_path = os.path.join(args.parent_dir, domain_name, f"PatchCoord/{isample[1]['pathology_id']}.h5")
            instance_info = get_instance_info(
                embedding_path, coord_path, p_idx=p_idx, OS_time=OS_time,
                events=events, Domain_id=domain_id, split_list=None,
                use_patch_nums=args.use_patch_nums,
                link_edge=args.link
            )

            for ifold in range(5):
                if isample[1]["patient_id"] in all_Gid_PID_maps[ifold]['train']:
                    train_graphs[f"split_{ifold}"].append(instance_info)
                elif isample[1]["patient_id"] in all_Gid_PID_maps[ifold]['test']:
                    test_graphs[f"split_{ifold}"].append(instance_info)
                else:
                    print(f"unknow set of {isample[1]['patient_id']}")
        for ifold in range(5):
            # print(f"{len(domain_feats)} feats in {json_file}")
            print(f"train: {len(train_graphs[f'split_{ifold}'])}, test: {len(test_graphs[f'split_{ifold}'])} graphs in {domain_name}")
            torch.save(train_graphs[f'split_{ifold}'], os.path.join(graph_file_saved_path, f'{domain_name}_train_{ifold}.pt'))
            torch.save(test_graphs[f'split_{ifold}'], os.path.join(graph_file_saved_path, f'{domain_name}_test_{ifold}.pt'))



def get_params():
    parser = argparse.ArgumentParser(description='superpixel_generate')

    parser.add_argument('--parent_dir', type=str, default='../dataset/TCGA')
    parser.add_argument('--json_dir', type=str, default='../dataset/TCGA')
    parser.add_argument('--graph_file_saved_path', type=str, default='../dataset/TCGA') # GraphData_random_link, GraphData
    # parser.add_argument('--norm_stain', nargs='+', default=["None", 'Vahadane', "Reinhard", "Ruifrok", "Macenko"]) # None, Reinhard, Ruifrok Macenko Vahadane
    parser.add_argument('--norm_stain', nargs='+', default=["None"]) # None, Reinhard, Ruifrok Macenko Vahadane
    parser.add_argument('--link', type=str, default='near8') # random, near8

    parser.add_argument('--use_patch_nums', type=int, default=-1)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    tcga_data = {
        # 5: 'brca',
        6: 'lgg',
    }

    try:
        args=get_params()
        main(args)

    except Exception as exception:
#         logger.exception(exception)
        raise