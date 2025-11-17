import os, random, torch
import sys

from numpy.ma.core import argmax

sys.path.append('.')
sys.path.append('../')


import numpy as np
import time
import argparse
import copy
from losses import E1TimeFitLoss
from Prognostic.utils.block_utils import get_split_list, setup_seed, visualize_tsne, check_all_same
from torchsurv.loss import cox
import pandas as pd
from torchsurv.metrics.cindex import ConcordanceIndex
from early_stopping_pytorch import EarlyStopping
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use('Agg')
setup_seed(42)

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# torch.use_deterministic_algorithms(True)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from model.GNNModel import GNNModel

from collections import defaultdict, OrderedDict
from sklearn.model_selection import train_test_split
import json
from torch_geometric.loader import DataLoader
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


def get_train_val_test_data(data_trainlists, data_testlists, train_all, graph_file_saved_path):
    train_set_name = data_trainlists
    test_set_name = data_testlists

    if train_all == 1.0:
        train_pyg_data = []
        val_pyg_data = []
        for itrain in train_set_name:
            pyg_data = torch.load(os.path.join(graph_file_saved_path, itrain), weights_only=False)
            train_pyg_data.extend(pyg_data)
            val_pyg_data.append(pyg_data)
    else:
        train_pyg_data = []
        val_pyg_data = []
        sub_pyg_data = []
        for itrain in train_set_name:
            pyg_data = torch.load(os.path.join(graph_file_saved_path, itrain), weights_only=False)
            X_train, X_test, y_train, y_test = train_test_split(
                pyg_data,
                [_ for _ in range(len(pyg_data))],
                test_size=1-train_all, shuffle=True, random_state=42
            )
            train_pyg_data.extend(X_train)
            sub_pyg_data.extend(X_test)
            print(f"len of train {len(X_train)} in {itrain}")
            print(f"len of val {len(X_test)} in {itrain}")
        val_pyg_data.append(sub_pyg_data)
        random.shuffle(train_pyg_data)


    test_pyg_data = []
    for itest in test_set_name:
        pyg_data = torch.load(os.path.join(graph_file_saved_path, itest), weights_only=False)
        test_pyg_data.append(pyg_data)

    return train_pyg_data, val_pyg_data, test_pyg_data


# ======================================================================================== #

def train_model(
        n_epochs, model, optimizer,
        train_set, val_set, test_set,
        batch_size, sampler_strategy,
        log_save_path, save_model_dir, args, cox_loss_weight, time_loss_weight, ifold=0
):
    best_ci = 0.
    bestVal_tr_ci = 0.
    best_epoch = 0
    selected_val_loss = 0

    criterion_time_fit = E1TimeFitLoss()

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(n_epochs):
        model.train()

        total_loss_for_train = 0

        sampler_lists, sampler_idx_lists = batch_sampler(
            len(train_set), sampler_strategy, batch_size
        )
        print(f"Epoch: {epoch}")
        loss_array = {}
        batches_log_info = ""
        for batch_ids, (sub_sample_list, sub_sample_idx_list) in enumerate(zip(sampler_lists, sampler_idx_lists)):
            batch_pyg = [train_set[_] for _ in sub_sample_list]
            batch = Batch.from_data_list(batch_pyg)
            batch_log_info = ""
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch)

            if "cox" in args.loss_name:
                if torch.sum(batch.events) > 0.0:
                    loss_neg = cox.neg_partial_log_likelihood(
                        output['os_time'], batch.events.bool().squeeze(),
                        batch.y.squeeze(), reduction="mean"
                    )
                    loss_array['cox'] = cox_loss_weight * loss_neg

            if "time_fit" in args.loss_name:
                if torch.sum(batch.events) > 0.0:
                    time_loss = criterion_time_fit(
                        output['time_fit'].squeeze(), batch.events.bool().squeeze(),
                        batch.y.squeeze()
                    )
                    loss_array['time_loss'] = time_loss_weight * time_loss

            loss = torch.tensor(0.0).to(device)
            for iloss_name, iloss in loss_array.items():
                loss += iloss
                batch_log_info += f"{iloss_name}:{iloss.item():6.3f}; "

            if torch.sum(batch.events) == 0.0:
                continue

            loss.backward()
            optimizer.step()
            batch_log_info += f"Batch Loss:{loss.item(): 5.3f}"
            batch_log_info += "\n"
            total_loss_for_train += loss.item()
            # if batch_ids % 1 == 0:
            #     print(batch_log_info)
            batches_log_info += batch_log_info

        #================================== train ==================================#
        tr_out_pre, tr_labelall_time, tr_labelall_surv_type, _ = infer_model(model, train_set)
        cindex = ConcordanceIndex()
        c_idx_for_train = cindex(
            tr_out_pre,
            tr_labelall_surv_type,
            tr_labelall_time
        )
        log_info = f"\ntrain ci: {c_idx_for_train:6.3f};\n"

        #================================== val ==================================#
        val_c_idxs = []
        val_c_loss = []
        for ival_slide in val_set:
            v_out_pre, v_labelall_time, v_labelall_surv_type, _ = infer_model(model, ival_slide)
            val_c_idx = cindex(
                v_out_pre,
                v_labelall_surv_type,
                v_labelall_time
            )
            val_loss_neg = cox.neg_partial_log_likelihood(
                v_out_pre, v_labelall_surv_type.bool().squeeze(),
                v_labelall_time.squeeze(), reduction="mean"
            )
            val_time_loss = criterion_time_fit(
                _['time_fit'].squeeze(), v_labelall_surv_type.bool().squeeze(),
                v_labelall_time.squeeze()
            )
            val_c_loss.append(cox_loss_weight * val_loss_neg.item() + time_loss_weight * val_time_loss.item())
            val_c_idxs.append(val_c_idx.cpu().numpy())

        for idx, ival_loss in enumerate(val_c_loss):
            log_info += f"val loss: {ival_loss:6.3f};\n"
        for idx, ival_ci in enumerate(val_c_idxs):
            log_info += f"val ci: {ival_ci:6.3f};\n"

        #================================== test ==================================#
        test_c_idxs = []
        for itest_slide in test_set:
            t_out_pre, t_labelall_time, t_labelall_surv_type, _ = infer_model(model, itest_slide)
            test_c_idx = cindex(
                t_out_pre,
                t_labelall_surv_type,
                t_labelall_time
            )
            test_c_idxs.append(test_c_idx.cpu().numpy())

        log_info = f"\ntrain ci: {c_idx_for_train:6.3f};\n"

        for idx, ival_ci in enumerate(val_c_idxs):
            log_info += f"val ci: {ival_ci:6.3f};\n"

        for idx, ival_loss in enumerate(val_c_loss):
            log_info += f"val loss: {ival_loss:6.3f};\n"

        for idx, itest_ci in enumerate(test_c_idxs):
            log_info += f"test {args.data_testlists[idx]}-th ci: {itest_ci:6.3f};\n"

        # print(log_info)
        with open(log_save_path, mode="a") as f:
            f.write(f"Epoch: {epoch} \n")
            f.write(batches_log_info)
            f.write(log_info)
            f.write("=" * 20 + "\n")

        c_model = copy.deepcopy(model)
        if np.mean(val_c_idxs) >= best_ci:
            best_ci = np.mean(val_c_idxs)
            bestVal_tr_ci = c_idx_for_train
            cur_test_ci = np.mean(test_c_idxs)
            cur_test_ci_detail = copy.deepcopy(test_c_idxs)
            best_epoch = epoch
            selected_val_loss = np.mean(val_c_loss)
            v_model = copy.deepcopy(model)

        print(f"At Epoch {epoch}: Test CI: {np.mean(test_c_idxs):6.3f}\n")

        print(f"At Epoch {best_epoch}: Best Val CI: {best_ci:6.3f}, with train ci: {bestVal_tr_ci:6.3f}, test ci {cur_test_ci:6.3f}, val loss {selected_val_loss:6.3f}\n")

        if epoch % 1 == 0:
            save_path = os.path.join(save_model_dir, f'saved_model_{ifold}.pth')
            torch.save({'v_model_state_dict': v_model.state_dict(),
                        'model_state_dict': c_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': best_epoch,
                        }, save_path)

        early_stopping(np.mean(val_c_loss), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(f"At Epoch {best_epoch}: Best Val CI: {best_ci:6.3f}, with test ci {cur_test_ci:6.3f}\n")
    for idx, itest_ci in enumerate(cur_test_ci_detail):
        print(f"test {args.data_testlists[idx]}-th ci: {itest_ci:6.3f};\n")

    with open(log_save_path, mode="a") as f:
        f.write("+" * 20 + "\n")
        f.write(f"At Epoch {best_epoch}: with test ci {cur_test_ci:6.3f}\n")
        for idx, itest_ci in enumerate(cur_test_ci_detail):
            f.write(f"test {args.data_testlists[idx]}-th ci: {itest_ci:6.3f};\n")
        f.write("#" * 20 + "\n")



    return_value = {
        "train_CI": bestVal_tr_ci,
        "val_CI": best_ci,
        "meanTest_CI": cur_test_ci,
    }
    for idx, itest_ci in enumerate(cur_test_ci_detail):
        return_value[f"{args.data_testlists[idx]}_CI"] = itest_ci
    return return_value

def infer_model(model, data_list):
    model.eval()
    with torch.no_grad():
        data_batch = Batch.from_data_list(data_list)
        outs = model(data_batch.to(device))

        t_out_pre = outs['os_time']
        t_labelall_surv_type = data_batch.events.bool()
        t_labelall_time = data_batch.y

    return t_out_pre, t_labelall_time.squeeze(), t_labelall_surv_type.squeeze(), outs


def main(args):

    n_epochs = args.epochs
    setup_seed(42)
    train_set, val_set, test_set = get_train_val_test_data(
        args.data_trainlists, args.data_testlists,
        args.train_all, os.path.join(args.parent_dir, args.graph_file_saved_path)
    )

    run_time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_result_dir = f"{args.log_result_dir}/{run_time_str}"
    os.makedirs(log_result_dir, exist_ok=True)
    config_save_path = os.path.join(log_result_dir, f'conf.log')

    with open(config_save_path, "w") as f:  # 设置文件对象
        for i in vars(args):
            f.write(i + ":" + str(vars(args)[i]) + '\n')
    f.close()
    print(f"saving successfully in {config_save_path}")

    exp_info = []
    exp_valinfo = []
    exp_traininfo = []
    for exp_no in range(args.exp_times):
        print(f"=============================== running time: {exp_no} ===========================================")
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

        optimizer = torch.optim.AdamW(
            [dict(params=model.parameters(), lr=args.lr, betas=(args.beta_low, args.beta_high), weight_decay=args.l2_reg_alpha), ])


        log_save_path = os.path.join(log_result_dir, f'run_{exp_no}.log')
        exp_res = train_model(
            n_epochs, model, optimizer,
            train_set, val_set, test_set,
            args.batch_size,
            args.sampler_strategy,
            log_save_path, log_result_dir,
            args, args.cox_loss_weight, args.time_loss_weight, ifold=exp_no
        )
        exp_traininfo.append(round(exp_res['train_CI'].item(), 4))
        exp_valinfo.append(round(exp_res['val_CI'], 4))
        exp_info.append(exp_res)
    print(f"===================== {args.exp_times} Exps done =============================")
    print(f"train: {exp_traininfo}")
    print(f"mean: {np.mean(exp_traininfo)}, std: {np.std(exp_traininfo)}\n")

    print(f"val: {exp_valinfo}")
    print(f"mean: {np.mean(exp_valinfo)}, std: {np.std(exp_valinfo)}\n")

    print(f"The best exp {argmax(exp_valinfo)} with {max(exp_valinfo):6.3f}")
    for ikey, ivalue in exp_info[argmax(exp_valinfo)].items():
        print(f"\t {ikey}: {ivalue:6.3f}")




def get_params():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--parent_dir', type=str, default='../dataset/GraphDataNone')
    parser.add_argument('--graph_file_saved_path', type=str, default='../dataset/GraphDataNone')
    parser.add_argument('--log_result_dir', type=str, default='../logs/log_result')

    # use for
    parser.add_argument("--train_all",  type=float, default=0.8)
    parser.add_argument("--infer_flag", action='store_true', default=False)
    parser.add_argument('--data_trainlists', nargs='+',
                        default=['rfs_huashan.pt', 'rfs_hz.pt'])
    parser.add_argument('--data_testlists', nargs='+', default=[
        'rfs_YouAn.pt', "rfs_huashan2.pt", 'rfs_CY.pt',
        # "rfs_huashan.pt", "rfs_hz.pt",
    ])
    parser.add_argument('--data_inferlists', nargs='+', default=[
        "rfs_huashan.pt", 'rfs_hz.pt', 'rfs_YouAn.pt', "rfs_huashan2.pt", 'rfs_CY.pt'
    ])

    parser.add_argument('--finished_model', type=str, default="")
    parser.add_argument('--vis_tsne', action='store_true', default=False)
    parser.add_argument('--PostProcess', default="AverageGFeat")
    ## model config
    parser.add_argument('--model_name', type=str, default='GNNModel') # GNNModel, AutoMerge
    # parser.add_argument('--loss_name', nargs='+', default=['cox'])
    parser.add_argument('--loss_name', nargs='+', default=['cox', "time_fit"])
    parser.add_argument('--gnn_intra', nargs='+', default=['gin', 'gin', 'gin', 'gat'])  # 'sage''TransformerConv'

    # GCN
    parser.add_argument('--mpool_inter', type=str,
                        default='mean')  # ‘global_mean_pool’,'global_max_pool','global_att_pool'

    parser.add_argument('--use_gnn_norm', action="store_true")
    #

    parser.add_argument('--exp_times', type=int, default=100)
    parser.add_argument('--in_feats_intra', type=int, default=512)
    parser.add_argument('--n_hidden_intra', type=int, default=1024)
    parser.add_argument('--out_feats_intra', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--drop_out_ratio', type=float, default=0.1)
    parser.add_argument('--IN_Ratio', type=float, default=0.7)
    # parser.add_argument('--T0', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)

    # train strategy
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument("--lr", type=float, default=6e-05, help="Learning rate of model training")
    parser.add_argument("--l2_reg_alpha", type=float, default=7.5e-06)
    parser.add_argument("--beta_low", type=float, default=0.9)
    parser.add_argument("--beta_high", type=float, default=0.99)

    parser.add_argument('--sampler_strategy', type=str, default="SequenceSampler")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--cox_loss_weight", type=float, default=2)
    parser.add_argument("--time_loss_weight", type=float, default=0.6)


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
