import os, random, torch
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')


import numpy as np
import time
import argparse
import copy
from losses import E1TimeFitLoss
from Prognostic.utils.block_utils import get_split_list, setup_seed, visualize_tsne, check_all_same
from torchsurv.loss import cox
import pandas as pd
from torchsurv.metrics.cindex import ConcordanceIndex
from model.GNNModel import GNNModel

from sklearn.model_selection import train_test_split
import json
from torch_geometric.data.batch import Batch

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


def batch_sampler(data_nums, strategy, batch_size):
    '''
    cid_per_batch: 每个batch选取多少个cid
    '''
    sampler_lists = []
    sampler_idx_lists = []

    if strategy == "SequenceSampler":
        num_iter = data_nums // batch_size + 1

        for i in range(num_iter):
            if i + 1 == num_iter:
                if i * batch_size < data_nums:
                    sampler_lists.append([_ for _ in range(i * batch_size, data_nums)])
                    sampler_idx_lists.append([_ for _ in range(i * batch_size, data_nums)])
            else:
                sampler_lists.append([_ for _ in range(i * batch_size, (i + 1) * batch_size)])
                sampler_idx_lists.append([_ for _ in range(i * batch_size, (i + 1) * batch_size)])
    else:
        raise "Unknown sampler strategy"

    return sampler_lists, sampler_idx_lists

# ======================================================================================== #

def infer_model(model, data_list):
    model.eval()
    with torch.no_grad():
        data_batch = Batch.from_data_list(data_list)
        outs = model(data_batch.to(device))

        t_out_pre = outs['os_time']
        t_labelall_surv_type = data_batch.events.bool()
        t_labelall_time = data_batch.y


    return t_out_pre, t_labelall_time.squeeze(1), t_labelall_surv_type.squeeze(1), outs


def main(args):

    cindex = ConcordanceIndex()

    if args.model_name == "GNNModel":
        model = GNNModel(in_feats_intra=args.in_feats_intra,
                         n_hidden_intra=args.n_hidden_intra,
                         out_feats_intra=args.out_feats_intra,
                         gnn_intra=args.gnn_intra,
                         num_heads=args.num_heads,
                         drop_out_ratio=args.drop_out_ratio,
                         mpool_inter=args.mpool_inter,
                         IN_Ratio=args.IN_Ratio,
                         use_gnn_norm=args.use_gnn_norm
                         )

    print(model)
    model = model.to(device)

    ema_weights_state = torch.load(args.finished_model)['v_model_state_dict']
    model.load_state_dict(
        ema_weights_state, strict=True
    )
    log_out_dir = args.out_result_dir
    os.makedirs(args.out_result_dir, exist_ok=True)
    run_time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    cindex_outfile = os.path.join(log_out_dir, f"CIndex_{run_time_str}.txt")

    output_str = f"load model: {args.finished_model}\n"

    print("Test on each inference dataset:")
    for domain_id, idata in enumerate(args.data_inferlists):
        pyg_data = torch.load(os.path.join(args.parent_dir, args.graph_file_saved_path, f"rfs_{idata}.pt"), weights_only=False)
        domain_name = idata.split(".pt")[0]
        print(f"Processing json data {domain_name} with {len(pyg_data)} WSI.")
        with open(os.path.join(args.parent_dir, args.graph_file_saved_path, f"rfs_{domain_name}_GP_maps.json"), 'r', encoding='utf-8') as load_f:
            ijson_data = json.load(load_f)

        # ================================================================================= #

        t_out_pre, t_labelall_time, t_labelall_surv_type, model_output = infer_model(model, pyg_data)
        try:
            test_c_idx = cindex(t_out_pre, t_labelall_surv_type, t_labelall_time)
            print(f"The C-Index of {idata}: {test_c_idx.item()}")
        except Exception as e:
            test_c_idx = -1
        predict_outfile = os.path.join(log_out_dir, f"{domain_name}.csv")

        if args.PostProcess == "None":
            print(f"no post process for each patient")
            predict_out = {
                'predict': [], 'label_time': [], 'event': [], 'PID': [], 'name': [],
            }
            for iout, irfs_time, ievent, ipyg_data in zip(
                    t_out_pre.cpu().numpy(), t_labelall_time.cpu().numpy(),
                    t_labelall_surv_type.cpu().numpy(), pyg_data
            ):
                patient_info = ijson_data[str(int(ipyg_data.graph_id))]
                predict_out['predict'].append(iout[0])
                predict_out['label_time'].append(irfs_time)
                predict_out['event'].append(ievent)
                predict_out['PID'].append(patient_info['PID'])
                predict_out['name'].append(patient_info['name'])
        elif args.PostProcess == "AverageGFeat":
            # get all patient name:
            print("merge features")
            predict_out = {
                'predict_GFeat_Avg': [],
                # 'predict_RFS_Avg': [],
                'label_time': [],
                'event': [], 'name': [],
            }
            patient_name2id = {}

            for iout, ig_feat, irfs_time, ievent, ipyg_data in zip(
                    t_out_pre.cpu().numpy(), model_output['g_feat'], t_labelall_time.cpu().numpy(),
                    t_labelall_surv_type.cpu().numpy(), pyg_data
            ):
                patient_info = ijson_data[str(int(ipyg_data.graph_id))]
                name = patient_info['name']
                if name not in patient_name2id:
                    patient_name2id[name] = {
                        "gfeats": [],
                        "predict": [],
                        "label_time": [],
                        "event": [],
                    }
                patient_name2id[name]['predict'].append(iout[0])
                patient_name2id[name]['label_time'].append(irfs_time)
                patient_name2id[name]['event'].append(ievent)
                patient_name2id[name]['gfeats'].append(ig_feat.unsqueeze(0))

            for iname, merge_info in patient_name2id.items():
                # predict_out['predict'].append(iname)
                model.eval()
                with torch.no_grad():
                    cat_tensor = torch.cat(merge_info['gfeats'], dim=0)
                    predict_out['predict_GFeat_Avg'].append(
                        torch.sigmoid(model.os_head(torch.mean(cat_tensor, dim=0))).cpu().numpy()[0]
                    )
                assert check_all_same(merge_info['event']), "multiple value of event"
                assert check_all_same(merge_info['label_time']), "multiple value of label_time"
                predict_out['name'].append(iname)
                # predict_out['predict_RFS_Avg'].append(np.mean(merge_info['predict']))
                predict_out['event'].append(np.mean(merge_info['event']))
                predict_out['label_time'].append(np.mean(merge_info['label_time']))

        pd.DataFrame(predict_out).to_csv(predict_outfile, encoding="utf_8_sig")
        output_str += f"Test on {domain_name}: {test_c_idx:6.3f}\n"+"=" * 20 + "\n"
    #
    print(f"Save CIndex to {cindex_outfile}")
    with open(cindex_outfile, mode="w+") as f:
        f.write(output_str)



def get_params():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--parent_dir', type=str, default='../dataset/GraphDataNone')
    parser.add_argument('--graph_file_saved_path', type=str, default='GraphData')
    parser.add_argument('--out_result_dir', type=str, default='../logs/log_result')

    # use for
    parser.add_argument("--infer_flag", action='store_true', default=False)
    parser.add_argument('--data_trainlists', nargs='+',
                        default=['rfs_huashan.pt', 'rfs_hz.pt'])
    parser.add_argument('--data_testlists', nargs='+', default=[
        'rfs_YouAn.pt', "rfs_huashan2.pt", 'rfs_CY.pt',
    ])
    parser.add_argument('--data_inferlists', nargs='+', default=[
        "rfs_huashan.pt", 'rfs_hz.pt', 'rfs_YouAn.pt', "rfs_huashan2.pt", 'rfs_CY.pt',
        "rfs_JiangData32.pt",
        "rfs_ZY4HE26605ZHL.pt", "rfs_file_202051101.pt"
    ])

    parser.add_argument('--finished_model', type=str, default="")
    parser.add_argument('--vis_tsne', action='store_true', default=False)
    parser.add_argument('--PostProcess', default="AverageGFeat")
    ## model config
    parser.add_argument('--model_name', type=str, default='GNNModel') # GNNModel, AutoMerge
    # parser.add_argument('--loss_name', nargs='+', default=['cox'])
    parser.add_argument('--loss_name', nargs='+', default=['cox', "time_fit"])
    parser.add_argument('--gnn_intra', nargs='+', default=['gin', 'gin', 'gin',  'gat'])  # 'sage''TransformerConv'

    # GCN
    parser.add_argument('--mpool_inter', type=str,
                        default='mean')  # ‘global_mean_pool’,'global_max_pool','global_att_pool'

    parser.add_argument('--use_gnn_norm', action="store_true")
    #

    parser.add_argument('--in_feats_intra', type=int, default=512)
    parser.add_argument('--n_hidden_intra', type=int, default=1024)
    parser.add_argument('--out_feats_intra', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--drop_out_ratio', type=float, default=0.1)
    parser.add_argument('--IN_Ratio', type=float, default=0.7)
    # parser.add_argument('--T0', type=int, default=5)

    # train strategy
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--sampler_strategy', type=str, default="SequenceSampler")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument("--cox_loss_weight", type=float, default=2)
    parser.add_argument("--time_loss_weight", type=float, default=0.6)
    parser.add_argument("--train_all",  type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=6.6e-05, help="Learning rate of model training")
    parser.add_argument("--l2_reg_alpha", type=float, default=7.5e-06)
    parser.add_argument("--beta_low", type=float, default=0.9)
    parser.add_argument("--beta_high", type=float, default=0.99)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    try:
        args = get_params()
        main(args)
    except Exception as exception:
        raise

    # n = 64
    # x = torch.randn((n, 16))
    # model_cox = torch.nn.Sequential(torch.nn.Linear(16, 1))
    # log_hz = model_cox(x)
    # event = torch.randint(low=0, high=2, size=(n,)).bool()
    # time = torch.randint(low=1, high=100, size=(n,)).float()
    # print(log_hz)
    # print(event)
    # print(time)
    # # loss = cox.neg_partial_log_likelihood(log_hz, event, time, reduction="mean")
    # cindex = ConcordanceIndex()
    # print(cindex(log_hz, event, time))