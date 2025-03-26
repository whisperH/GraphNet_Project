import os, random, torch
import sys
sys.path.append('.')
sys.path.append('../')

import numpy as np
import time
import argparse
import copy
from losses import E1TimeFitLoss
from utils.block_utils import get_split_list, setup_seed, visualize_tsne, check_all_same
# https://github.com/lucidrains/ema-pytorch
from ema_pytorch import PostHocEMA
from torchsurv.loss import cox
import pandas as pd
from torchsurv.metrics.cindex import ConcordanceIndex

import matplotlib
matplotlib.use('TkAgg')
setup_seed(2)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# torch.use_deterministic_algorithms(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from lifelines.utils import concordance_index as ci
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


def batch_sampler(data_info, data_nums, strategy, epoch, batch_size, instance_per_cid=4, split_num=500):
    '''
    cid_per_batch: 每个batch选取多少个cid
    '''
    sampler_lists = []
    sampler_label_lists = []
    cid_per_batch = batch_size // instance_per_cid

    if strategy == "SequenceSampler":
        num_iter = data_nums // batch_size + 1

        for i in range(num_iter):
            if i + 1 == num_iter:
                if i * batch_size < data_nums:
                    sampler_lists.append([_ for _ in range(i * batch_size, data_nums)])
                    sampler_label_lists.append([_ for _ in range(i * batch_size, data_nums)])
            else:
                sampler_lists.append([_ for _ in range(i * batch_size, (i + 1) * batch_size)])
                sampler_label_lists.append([_ for _ in range(i * batch_size, (i + 1) * batch_size)])
    else:
        raise "Unknown sampler strategy"

    return sampler_lists, sampler_label_lists


def get_train_val_test_data(args):
    train_set_name = args.data_trainlists
    # if args.data_vallists is None:
    #     val_set_name = train_set_name
    # else:
    #     val_set_name = args.data_vallists

    test_set_name = args.data_testlists

    train_pyg_data = []
    val_pyg_data = []
    sub_pyg_data = []
    if args.train_all == 1.0:
        for itrain in train_set_name:
            pyg_data = torch.load(os.path.join(args.graph_file_saved_path, itrain))

            train_pyg_data.extend(pyg_data)
            val_pyg_data.append(pyg_data)
    else:
        for itrain in train_set_name:
            pyg_data = torch.load(os.path.join(args.graph_file_saved_path, itrain))
            X_train, X_test, y_train, y_test = train_test_split(
                pyg_data,
                [_ for _ in range(len(pyg_data))],
                test_size=1-args.train_all, shuffle=True, random_state=1
            )
            train_pyg_data.extend(X_train)
            sub_pyg_data.extend(X_test)
        val_pyg_data.append(sub_pyg_data)
    random.shuffle(train_pyg_data)

    print(f"len of {len(train_pyg_data)}")

    # for ival in val_set_name:
    #     pyg_data = torch.load(os.path.join(args.graph_file_saved_path, ival))
    #     val_pyg_data.append(pyg_data)

    test_pyg_data = []
    for itest in test_set_name:
        pyg_data = torch.load(os.path.join(args.graph_file_saved_path, itest))
        test_pyg_data.append(pyg_data)

    E0_domain2cid = defaultdict(list)
    E1_domain2cid = defaultdict(list)
    E0_cid = []
    E1_cid = []
    cid2index = defaultdict(list)
    Event_Count = {
        "event0": 0,
        "event1": 0,
    }
    DomainEvent_Count = {}
    domain_cid2indx = {}
    for data_index, pyd_data in enumerate(train_pyg_data):
        c_id = pyd_data.cluster_id
        domain_id = pyd_data.domain_id
        if int(domain_id) not in DomainEvent_Count:
            DomainEvent_Count[int(domain_id)] = {
                "event0": 0,
                "event1": 0,
            }
        if int(domain_id) not in domain_cid2indx:
            domain_cid2indx[int(domain_id)] = defaultdict(list)

        if int(pyd_data.events) == 0:
            E0_cid.append(int(c_id))
            if int(c_id) not in E0_domain2cid[int(domain_id)]:
                E0_domain2cid[int(domain_id)].append(int(c_id))

            DomainEvent_Count[int(domain_id)]['event0'] += 1
            Event_Count['event0'] += 1
        else:
            E1_cid.append(int(c_id))
            if int(c_id) not in E1_domain2cid[int(domain_id)]:
                E1_domain2cid[int(domain_id)].append(int(c_id))
            DomainEvent_Count[int(domain_id)]['event1'] += 1
            Event_Count['event1'] += 1

        domain_cid2indx[int(domain_id)][int(c_id)].append(data_index)
        cid2index[int(c_id)].append(data_index)

    cids = sorted(list(cid2index.keys()))
    domains = list(set(list(sorted(E0_domain2cid.keys())) + list(sorted(E1_domain2cid.keys()))))
    num_cids = len(cids)
    cid2label = {_: idx for idx, _ in enumerate(cids)}  # cids中的类别标签是稀疏的，无法直接用于分类
    # 如果不考虑domain 仅考虑cluster id：直接在cid2index中进行选取 数据索引
    # 如果考虑domain 和 cluster id：先在 domain2cid 中选取，对应的cid，再在 cid2index 中选取数据索引
    data_info = {
        "E0_domain2cid": E0_domain2cid,  # 对应domain id中的 cluster id
        "E0_cid": list(set(E0_cid)),  # 对应domain id中的 cluster id
        "E1_domain2cid": E1_domain2cid,  # 对应domain id中的 cluster id
        "E1_cid": list(set(E1_cid)),  # 对应domain id中的 cluster id
        "DomainEvent_Count": DomainEvent_Count,  # 对应domain id中的 cluster id
        "Event_Count": Event_Count,  # 对应domain id中的 cluster id
        "cid2index": cid2index,  # cluster id 对应哪些数据索引 on all
        "domain_cid2indx": domain_cid2indx,  # cluster id 对应哪些数据索引 with domain
        "cids": cids,  # cluster id 的列表
        "domains": domains,  # domain id 的列表
        "num_cids": num_cids,  # cluster id 的数量
        "cid2label": cid2label,  # 用于训练的cid类别名称
    }

    return train_pyg_data, val_pyg_data, test_pyg_data, data_info


# ======================================================================================== #

def train_model(
        n_epochs, model, optimizer,
        train_set, val_set, test_set,
        batch_size, data_info, sampler_strategy,
        log_save_path, save_model_dir, cindex
):
    best_ci = 0.
    best_test_ci = 0.
    best_epoch = 0
    best_test_epoch = 0

    split_list = get_split_list(args.split_num, args.ultra_split_num, args.ultra_start, args.ultra_end)
    split_len = len(split_list)
    criterion_graph_cls = torch.nn.CrossEntropyLoss()
    criterion_time_fit = E1TimeFitLoss()

    if args.use_ema:
        print("use ema for weight average")
        ema = PostHocEMA(
            model,
            sigma_rels=(0.1, 0.4),
            # a tuple with the hyperparameter for the multiple EMAs. you need at least 2 here to synthesize a new one
            update_every=1,  # how often to actually update, to save on compute (updates every 10th .update() call)
            checkpoint_every_num_steps=1,
            checkpoint_folder=f'{save_model_dir}/post-hoc-ema-checkpoints'
            # the folder of saved checkpoints for each sigma_rel (gamma) across timesteps with the hparam above, used to synthesizing a new EMA model after training
        )

    for epoch in range(n_epochs):
        model.train()

        total_loss_for_train = 0

        sampler_lists, sampler_label_lists = batch_sampler(
            data_info, len(train_set), sampler_strategy,
            epoch, batch_size, instance_per_cid=args.instance_per_cid,
            split_num=split_len
        )
        # print(sampler_lists)
        # print(sampler_label_lists)
        print(f"Epoch: {epoch}")
        loss_array = {}
        batches_log_info = ""
        for batch_ids, (sub_sample_list, sub_sample_label_list) in enumerate(zip(sampler_lists, sampler_label_lists)):
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
                    loss_array['cox'] = args.cox_loss_weight * loss_neg

            if "time_fit" in args.loss_name:
                if torch.sum(batch.events) > 0.0:
                    time_loss = criterion_time_fit(
                        output['time_fit'].squeeze(), batch.events.bool().squeeze(),
                        batch.y.squeeze()
                    )
                    loss_array['time_loss'] = args.time_loss_weight * time_loss

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
            if batch_ids % 10 == 0:
                print(batch_log_info)
            batches_log_info += batch_log_info

        tr_out_pre, tr_labelall_time, tr_labelall_surv_type, _ = infer_model(model, train_set)
        c_idx_for_train = cindex(
            tr_out_pre,
            tr_labelall_surv_type,
            tr_labelall_time
        )

        # val
        val_c_idxs = []
        for ival_slide in val_set:
            v_out_pre, v_labelall_time, v_labelall_surv_type, _ = infer_model(model, ival_slide)
            val_c_idx = cindex(
                v_out_pre,
                v_labelall_surv_type,
                v_labelall_time
            )
            val_c_idxs.append(val_c_idx.cpu().numpy())

        # test
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

        for idx, itest_ci in enumerate(test_c_idxs):
            log_info += f"test {args.data_testlists[idx]}-th ci: {itest_ci:6.3f};\n"

        print(log_info)
        with open(log_save_path, mode="a") as f:
            f.write(f"Epoch: {epoch} \n")
            f.write(batches_log_info)
            f.write(log_info)
            f.write("=" * 20 + "\n")

        if np.mean(val_c_idxs) >= args.ema_threshold and args.use_ema and epoch>args.ema_save_epoch:
            print("save pth for ema")
            ema.update()


        if np.mean(val_c_idxs) >= best_ci:
            best_ci = np.mean(val_c_idxs)
            cur_test_ci = np.mean(test_c_idxs)
            best_epoch = epoch
            v_model = copy.deepcopy(model)
        if np.mean(test_c_idxs) >= best_test_ci:
            best_test_epoch = epoch
            best_test_ci = np.mean(test_c_idxs)
            cur_val_ci = np.mean(val_c_idxs)
            t_model = copy.deepcopy(model)

        print(f"At Epoch {best_epoch}: Best Val CI: {best_ci:6.3f}, with test ci {cur_test_ci:6.3f}\n"
              f"At Epoch {best_test_epoch}: val ci {cur_val_ci:6.3f}, Best Test CI: {best_test_ci:6.3f}")

        if epoch % 2 == 0:
            save_path = os.path.join(save_model_dir, f'{epoch}_model.pth')
            torch.save({'v_model_state_dict': v_model.state_dict(),
                        't_model_state_dict': t_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, save_path)

    print(f"At Epoch {best_epoch}: Best Val CI: {best_ci:6.3f}\n")
    print(f"At Epoch {best_test_epoch}: Best test CI: {best_test_ci:6.3f}\n")

    with open(log_save_path, mode="a") as f:
        f.write(f"At Epoch {best_epoch}: Best Val CI: {best_ci:6.3f}, Saved model {save_path}\n")
        f.write(f"At Epoch {best_test_epoch}: Best Test CI: {best_test_ci:6.3f}, Saved model {save_path}\n")
        f.write("=" * 20 + "\n")

    if args.use_ema:
        synthesized_ema = ema.synthesize_ema_model(sigma_rel=0.05)
        val_c_idxs = []
        for ival_slide in val_set:
            v_out_pre, v_labelall_time, v_labelall_surv_type, _ = infer_model(synthesized_ema, ival_slide)
            val_c_idx = cindex(
                v_out_pre,
                v_labelall_surv_type,
                v_labelall_time
            )
            print(f"EMA on Val {val_c_idx}")
            val_c_idxs.append(val_c_idx.cpu().numpy())

        # test
        test_c_idxs = []
        for idx, itest_slide in enumerate(test_set):
            t_out_pre, t_labelall_time, t_labelall_surv_type, _ = infer_model(synthesized_ema, itest_slide)
            test_c_idx = cindex(
                t_out_pre,
                t_labelall_surv_type,
                t_labelall_time
            )
            print(f"EMA on {args.data_testlists[idx]}: {test_c_idx}")
            test_c_idxs.append(test_c_idx.cpu().numpy())

        print(f"EMA on Val {np.mean(val_c_idxs)}")
        print(f"EMA on Test {np.mean(test_c_idxs)}")
        with open(log_save_path, mode="a") as f:
            f.write(f"EMA Val CI: {np.mean(val_c_idxs):6.3f}\n")
            f.write(f"EMA Test CI: {np.mean(test_c_idxs):6.3f}\n")
            f.write("=" * 20 + "\n")

        save_path = os.path.join(save_model_dir, f'ema_model.pth')
        torch.save({'model_state_dict': synthesized_ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, save_path)

def infer_model(model, data_list):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        data_batch = Batch.from_data_list(data_list)
        outs = model(data_batch.to(device))

        t_out_pre = outs['os_time']
        t_labelall_surv_type = data_batch.events.bool()
        t_labelall_time = data_batch.y

        _, predicted = torch.max(outs['g_id'].data, 1)
        total += data_batch.domain_id.squeeze().size(0)
        correct += (predicted == data_batch.domain_id.squeeze()).sum().item()
    # print(f'The accuracy of Domain Classification: {correct / total}')

    return t_out_pre, t_labelall_time.squeeze(), t_labelall_surv_type.squeeze(), outs


def main(args):

    n_epochs = args.epochs
    batch_size = args.batch_size
    cindex = ConcordanceIndex()

    train_set, val_set, test_set, data_info = get_train_val_test_data(args)

    if args.model_name == "GNNModel":
        model = GNNModel(in_feats_intra=args.in_feats_intra,
                         n_hidden_intra=args.n_hidden_intra,
                         out_feats_intra=args.out_feats_intra,
                         gnn_intra=args.gnn_intra,
                         # domain_nums=len(data_info['domains']),
                         mpool_inter=args.mpool_inter,
                         use_gnn_norm=args.use_gnn_norm
                         )

    model = model.to(device)
    if not args.infer_flag:
        log_result_dir = f"{args.log_result_dir}/{run_time_str}"
        os.makedirs(log_result_dir, exist_ok=True)

        log_save_path = os.path.join(log_result_dir, f'run.log')
        config_save_path = os.path.join(log_result_dir, f'conf.log')

        with open(config_save_path, "w") as f:  # 设置文件对象
            for i in vars(args):
                f.write(i + ":" + str(vars(args)[i]) + '\n')
        f.close()
        print(f"saving successfully in {config_save_path}")

        optimizer = torch.optim.AdamW(
            [dict(params=model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.l2_reg_alpha), ])


        train_model(
            n_epochs, model, optimizer,
            train_set, val_set, test_set,
            batch_size, data_info,
            args.sampler_strategy,
            log_save_path, log_result_dir,
            cindex
        )
    else:
        ema_weights_state = torch.load(args.finished_model)['v_model_state_dict']
        ema_model_state = OrderedDict()
        model_state = model.state_dict()
        for k, v in ema_weights_state.items():
            name = k.replace("ema_model.", "")
            if name in model_state:
                ema_model_state[name] = v
            else:
                print(f"ignore the {name} in ema model, please check the necessary of this parameters")
        model.load_state_dict(
            ema_model_state, strict=True
        )
        log_out_dir = os.path.split(args.finished_model)[0]

        all_patch_feats = []
        all_mergepatch_feats = []
        all_graph_feats = []
        all_cid_label = []
        all_domain_label = []
        all_patch_domain_labels = []
        all_patch_cid_labels = []
        all_mergepatch_cid_labels = []

        cindex_outfile = os.path.join(log_out_dir, "CIndex.txt")

        output_str = f"load model: {args.finished_model}\n"

        print("Test on all train data:")
        pyg_train_data = []
        for tr_domain_id, tr_idata in enumerate(args.data_trainlists):
            pyg_data = torch.load(os.path.join(args.graph_file_saved_path, tr_idata))
            pyg_train_data.extend(pyg_data)
        tr_out_pre, tr_labelall_time, tr_labelall_surv_type, tr_model_output = infer_model(model, pyg_train_data)
        train_c_idx = cindex(
            tr_out_pre,
            tr_labelall_surv_type,
            tr_labelall_time
        )
        print(train_c_idx)
        output_str += f"Test on all train dataset: {train_c_idx:6.3f}\n" + "=" * 20 + "\n"

        print("Test on each test dataset:")
        for domain_id, idata in enumerate(args.data_inferlists):
            pyg_data = torch.load(os.path.join(args.graph_file_saved_path, idata))
            domain_name = idata.split(".pt")[0]
            print(f"Processing json data {domain_name}")
            domain_patch_cid_labels = []
            domain_mergepatch_cid_labels = []
            with open(os.path.join(args.graph_file_saved_path, f"{domain_name}_GP_maps.json"), 'r', encoding='utf-8') as load_f:
                ijson_data = json.load(load_f)

            # ================================================================================= #

            t_out_pre, t_labelall_time, t_labelall_surv_type, model_output = infer_model(model, pyg_data)
            try:
                test_c_idx = cindex(t_out_pre, t_labelall_surv_type, t_labelall_time)
                print(test_c_idx)
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
                    'predict_GFeat_Avg': [], 'predict_RFS_Avg': [], 'label_time': [],
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
                    predict_out['predict_RFS_Avg'].append(np.mean(merge_info['predict']))
                    predict_out['event'].append(np.mean(merge_info['event']))
                    predict_out['label_time'].append(np.mean(merge_info['label_time']))

            pd.DataFrame(predict_out).to_csv(predict_outfile, encoding="utf_8_sig")
            output_str += f"Test on {domain_name}: {test_c_idx:6.3f}\n"+"=" * 20 + "\n"
        #
        print(f"Save CIndex to {cindex_outfile}")
        with open(cindex_outfile, mode="w+") as f:
            f.write(output_str)


        if args.vis_tsne:
            all_cid_label.extend([int(_.cluster_id) for _ in pyg_data])
            all_graph_feats.extend(model_output["g_feat"].cpu().numpy())

            for ig in pyg_data:
                node_size = ig.x.size(0)
                domain_patch_cid_labels.extend([int(ig.cluster_id)] * node_size)
                domain_mergepatch_cid_labels.extend([int(ig.cluster_id)] * args.vw_num)

            all_domain_label.extend([domain_id] * len(ijson_data))
            all_mergepatch_feats.extend(
                model_output["merge_node_feat"].reshape(-1, args.feat_dim).cpu().numpy()[:4500, :])
            all_patch_feats.extend(model_output["node_feat"].cpu().numpy()[:1500, :])
            all_patch_domain_labels.extend([domain_id] * 1500)
            all_patch_cid_labels.extend(domain_patch_cid_labels[:1500])
            all_mergepatch_cid_labels.extend(domain_mergepatch_cid_labels[:4500])

            # visualize_tsne(np.array(all_patch_feats), np.array(all_patch_domain_labels), args.data_inferlists)
            # visualize_tsne(np.array(all_patch_feats), np.array(all_patch_cid_labels), np.array(all_cid_label))
            visualize_tsne(
                np.array(all_mergepatch_feats),
                np.array(all_mergepatch_cid_labels),
                np.array(all_cid_label)
            )
            # visualize_tsne(np.array(all_graph_feats), np.array(all_cid_label), np.array(all_cid_label))


def get_params():
    parser = argparse.ArgumentParser(description='model training')
    parser.add_argument('--graph_file_saved_path', type=str, default='../dataset/GraphDataNone')
    parser.add_argument('--log_result_dir', type=str, default='../logs/log_result')

    # use for
    parser.add_argument("--train_all",  type=float, default=0.8)
    parser.add_argument("--infer_flag", action='store_true', default=False)
    # parser.add_argument('--data_trainlists', nargs='+', default=['rfs_check_youan.pt'])
    parser.add_argument('--data_trainlists', nargs='+',
                        default=['rfs_huashan.pt', 'rfs_hz.pt'])
                        # default=['rfs_check_hz.pt'])
    parser.add_argument('--data_vallists', nargs='+', default=None)
    parser.add_argument('--data_testlists', nargs='+', default=[
        'rfs_YouAn.pt', "rfs_huashan2.pt", 'rfs_CY.pt',
        # "rfs_check_youan.pt", "rfs_huashan2.pt", "rfs_CY.pt",
    ])
    # parser.add_argument('--data_testlists', nargs='+', default=['rfs_check_cy.pt'])
    parser.add_argument('--data_inferlists', nargs='+', default=[
        "rfs_huashan.pt", 'rfs_hz.pt', 'rfs_YouAn.pt', "rfs_huashan2.pt", 'rfs_CY.pt',
        "rfs_JiangData32.pt"
    ])
    # parser.add_argument('--data_inferlists', nargs='+',
    #                     default=["rfs_new_cy.pt", "rfs_new_youan.pt"])
    parser.add_argument('--finished_model', type=str, default="../logs/log_result/EMA_GIN_ALL70_TimeFit/20_model.pth")
    parser.add_argument('--vis_tsne', action='store_true', default=False)
    parser.add_argument('--PostProcess', default="None")
    ## model config
    parser.add_argument('--model_name', type=str, default='GNNModel') # GNNModel, AutoMerge
    # parser.add_argument('--loss_name', nargs='+', default=['cox'])
    parser.add_argument('--loss_name', nargs='+', default=['cox', "time_fit"])
    parser.add_argument('--gnn_intra', nargs='+', default=['gin', 'gin', 'gin', 'gat'])  # 'sage''TransformerConv'

    # GCN
    parser.add_argument('--mpool_inter', type=str,
                        default='mean')  # ‘global_mean_pool’,'global_max_pool','global_att_pool'
    parser.add_argument('--merge_method', type=str,
                        default='weighted_sum')  # ‘global_mean_pool’,'global_max_pool','global_att_pool'
    parser.add_argument('--use_gnn_norm', default=True)
    #
    parser.add_argument('--final_fea_type', type=str, default='cat_weights_sum')

    parser.add_argument('--vw_num', type=int, default=4)
    parser.add_argument('--feat_dim', type=int, default=2048)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--in_feats_intra', type=int, default=512)
    parser.add_argument('--n_hidden_intra', type=int, default=1024)
    parser.add_argument('--out_feats_intra', type=int, default=1024)

    # train strategy
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--use_ema', action='store_true', default=False)
    parser.add_argument('--ema_threshold', type=float, default=0.590)
    parser.add_argument('--ema_save_epoch', type=float, default=10)
    parser.add_argument("--lr", type=float, default=7.2668e-06, help="Learning rate of model training")
    parser.add_argument("--l2_reg_alpha", type=float, default=2.077e-05)

    parser.add_argument('--sampler_strategy', type=str, default="SequenceSampler")
    # SequenceSampler RandomIdentitySampler NeighbourIdentitySampler MultiDomainE1NeighbourIdentitySampler
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument("--instance_per_cid", type=int, default=4)
    parser.add_argument('--split_num', type=float, default=500,
                        help="should be same with split_num in generate_instance_graph")
    parser.add_argument('--ultra_split_num', type=float, default=0)
    parser.add_argument('--ultra_start', type=float, default=9)
    parser.add_argument('--ultra_end', type=float, default=50)
    parser.add_argument("--win_size", type=int, default=3)



    parser.add_argument("--cox_loss_weight", type=float, default=9)
    parser.add_argument("--time_loss_weight", type=float, default=0.1)
    parser.add_argument("--assign_loss", type=float, default=1)
    parser.add_argument("--tri_loss", type=float, default=1)

    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--soft_thred", type=float, default=0.1)
    parser.add_argument("--use_marginList", type=bool, default=True)

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