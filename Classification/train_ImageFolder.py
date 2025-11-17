#                    _ooOoo_
#                   o8888888o
#                   88" . "88
#                   (| -_- |)
#                    O\ = /O
#                ____/`---'\____
#              .   ' \\| |// `.
#               / \\||| : |||// \
#             / _||||| -:- |||||- \
#               | | \\\ - /// | |
#             | \_| ''\---/'' | |
#              \ .-\__ `-` ___/-. /
#           ___`. .' /--.--\ `. . __
#        ."" '< `.___\_<|>_/___.' >'"".
#       | | : `- \`.;`\ _ /`;.`/ - ` : | |
#         \ \ `-. \_ __\ /__ _/ .-` / /
# ======`-.____`-.___\_____/___.-`____.-'======
#                    `=---='


import argparse
import json
import logging
from operator import mod
import os
import sys
import time
import yaml
import math

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
import numpy as np
torch.set_printoptions(threshold=np.inf)

sys.path.append('../../..')
sys.path.append('.')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from dataset.Colon_loader import ImageFolderDataset
from model import MODELS  # noqa
from tools.utils import ConfusionMatrix, load_tif_dict, load_pretrain_checkpoint
import torch.optim as optim

def train_epoch(summary_train, summary_writer, epoch, model, loss_fn, optimizer, dataloader_train, confusion):
    model.train()
    steps = len(dataloader_train)
    time_now = time.time()
    loss_sum = 0
    grid_size = (cfg["dataset"]["image_size"]//cfg["dataset"]["patch_size"])**2

    summary_train['epoch'] = epoch
    print("train steps:", steps)
    for step,tmp in enumerate(dataloader_train):
        time_now = time.time()
        data, target, _, _ = tmp
        data = data.to(device)
        target = target.to(device)
        output_tuple = model(data)
        target = target.view(-1)
        # output = output_tuple[0]
        output = output_tuple[:, cfg["model"]["use_level_to_cal"]["train"][0]*grid_size:(cfg["model"]["use_level_to_cal"]["train"][-1]+1)*grid_size, :]
        if cfg["model"]["num_classes"] == 1:
            output = output.contiguous().view(-1)
            if "GoogLeNetv3" in cfg["model"]["model_name"]:
                loss = loss_fn(output, target) + 0.3 * loss_fn(output_tuple[2], target)
            else:
                loss = loss_fn(output, target)
        else:
            output = output.view(len(target), -1)
            loss = loss_fn(output, target.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_data = loss.item()
        loss_sum += loss_data
        confusion.update(output.clone().detach().cpu(), target.clone().detach().cpu().numpy())
        time_spent = time.time() - time_now
        logging.info(
            '{}, Epoch : {}, Step : {}, lr: {}, Training Loss: {:.5f}, '
            ' Run Time : {:.2f}'
                .format(
                time.strftime("%Y-%m-%d %H:%M:%S"), summary_train['epoch'],
                                                    summary_train['step'], optimizer.param_groups[0]["lr"], loss_data, time_spent))

        summary_train['step'] += 1
        # debug
        # if step == 2:
        #     break
    # print the result table of confusion
    summery_dic = confusion.summary()

    summary_train['loss'] = loss_sum / steps
    summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], epoch)   
    summary_writer.add_scalar('train/loss', summary_train['loss'], epoch)
    for k, v in summery_dic.items():
        summary_writer.add_scalar('train/'+k, v, epoch)

    print("summary_train: ", summary_train)

    return summary_train

def valid_epoch(summary_valid, summary_writer, epoch, model, loss_fn, dataloader_valid, confusion):
    model.eval()
    steps = len(dataloader_valid)
    batch_size = dataloader_valid.batch_size
    grid_size = dataloader_valid.dataset._grid_size
    dataiter = iter(dataloader_valid)
    loss_sum = 0
    summary_valid['epoch'] = epoch
    grid_size = (cfg["dataset"]["image_size"]//cfg["dataset"]["patch_size"])**2
    time_now = time.time()

    if epoch == cfg["train"]["epoch"] - 1:
        res_file = open(os.path.join(
            cfg["saver"]["ckpt_save_path"], cfg["saver"]["experiment_name"],
            cfg["saver"]["test_result"]
        ), "w", encoding="utf-8")
        os.makedirs(os.path.join(ckpt_save_path, "draw_hard_case"), exist_ok=True)

    print("valid steps:", steps)
    with torch.no_grad():
        for step in range(steps):
            time_now = time.time()
            data, target, pid, img_name= next(dataiter)
            data = data.to(device)
            target = target.to(device)
            output_tuple = model(data)

            target = target.view(-1)
            output = output_tuple[:, cfg["model"]["use_level_to_cal"]["val"][0]*grid_size:(cfg["model"]["use_level_to_cal"]["val"][-1]+1)*grid_size, :]
            if cfg["model"]["num_classes"] == 1:
                output = output.contiguous().view(-1)
                if "GoogLeNetv3" in cfg["model"]["model_name"]:
                    loss = loss_fn(output, target) + 0.3 * loss_fn(output_tuple[2], target)
                else:
                    loss = loss_fn(output, target)
            else:
                output = output.view(len(target), -1)
                loss = loss_fn(output, target.long())
            loss_data = loss.item()
            loss_sum += loss_data
            confusion.update(output.clone().detach().cpu(), target.clone().detach().cpu().numpy())
            time_spent = time.time() - time_now
            logging.info(
            '{}, Epoch : {}, Step : {}, Valid Loss: {:.5f}, '
            ' Run Time : {:.2f}'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, 
                        summary_valid['step'], loss_data, time_spent))
            if epoch == cfg["train"]["epoch"] - 1:
                if cfg["model"]["num_classes"] == 1:
                    output = output.sigmoid()
                    preds = torch.zeros_like(target)
                    preds[output > cfg['val']['thresh']] = 1
                else:
                    preds = torch.max(output, dim=1)[1]
                for idx in range(len(target)):
                    res_dic = {}
                    res_dic["filename"] = img_name[idx//grid_size]
                    name_split = res_dic["filename"].split("_")
                    x_center, y_center = int(name_split[-2]), int(name_split[-1].strip(".png"))
                    res_dic["x_center"], res_dic["y_center"] = x_center, y_center
                    res_dic["grid_idx"] = idx % grid_size
                    res_dic["is_correct"] = (target[idx].item()==preds[idx].item())
                    res_dic["label"] = int(target[idx].item())
                    res_dic["pred"] = int(preds[idx].item())
                    res_file.write(json.dumps(res_dic)+"\n")
            # debug
            # if step == 2:
            #     break
            summary_valid['step'] += 1

    summery_dic = confusion.summary()
    summary_valid['loss'] = loss_sum / steps   
    summary_valid['acc'] = summery_dic['acc']
    summary_writer.add_scalar('valid/loss', summary_valid['loss'], epoch)
    for k, v in summery_dic.items():
        summary_writer.add_scalar('valid/'+k, v, epoch)

    print("summary_valid: ",summary_valid)
    
    return summary_valid

def run():
    logging.basicConfig(level=logging.INFO)
    cfg_dataset, cfg_train, cfg_valid, cfg_saver, cfg_model = cfg["dataset"], cfg["train"], cfg["val"], cfg["saver"], cfg["model"]

    num_GPU = len(cfg_train["device_ids"].split(','))
    batch_size_train = cfg_train['batch_size_perGPU'] * num_GPU
    batch_size_valid = cfg_valid['batch_size_perGPU'] * num_GPU
    num_workers = cfg_train["num_workers_perGPU"] * num_GPU


    image_size, patch_size, crop_size = cfg['dataset']['image_size'], cfg['dataset']['patch_size'], cfg['dataset']['crop_size']
    if image_size % patch_size != 0:
        raise Exception('Image size / patch size != 0 : {} / {}'.
                        format(image_size, patch_size))
    patch_per_side = image_size // patch_size
    grid_size = patch_per_side * patch_per_side
    
    model = MODELS[cfg_model['model_name']](num_classes=cfg_model['num_classes'], 
                                            num_nodes=grid_size, 
                                            use_crf=cfg_model['use_crf'],
                                            pretrained=cfg_model['pretrain'], 
                                            use_levels=cfg_dataset.get("use_levels", [0]),
                                            level_crf=cfg_model.get("level_crf", False),
                                            concat_level_feats=cfg_model.get("concat_level_feats", False),
                                            cfg=cfg)
    if cfg_train['pretrain'] != '':
        model = load_pretrain_checkpoint(cfg_train['pretrain'], model)
    if cfg_train['resume'] != '':
        print("Use ckpt: ", cfg_train['resume'])
        assert len(cfg_train['resume']) != 0, "Please input a valid resume ckpt_path"
        model_tmp = torch.load(cfg_train['resume'])
        model.load_state_dict(model_tmp["state_dict"])
        cfg_train["start_epoch"] = model_tmp['epoch']
    # model = DataParallel(model, device_ids=None)
    model = model.to(device)

    if cfg_train["loss_fn"]["name"] == "FocalLoss":
        from Classification.model.loss_fn import focal_loss
        loss_fn = focal_loss(alpha=cfg_train["loss_fn"]["kwargs"]["alpha"],
                            gamma=cfg_train["loss_fn"]["kwargs"]["gamma"],
                            num_classes=cfg_model["num_classes"]).to(device)
        print("use focal loss")
    elif cfg_train["loss_fn"]["name"] == "BCEFocalLoss":
        from Classification.model.loss_fn import BCEFocalLoss
        loss_fn = BCEFocalLoss(alpha=cfg_train["loss_fn"]["kwargs"]["alpha"],
                            gamma=cfg_train["loss_fn"]["kwargs"]["gamma"],
                            reduction=cfg_train["loss_fn"]["kwargs"]["reduction"]).to(device)
        print("use BCEFocalLoss loss")
    else:
        if cfg_model["num_classes"] == 1:
            loss_fn = torch.nn.BCEWithLogitsLoss().to(device)
            print("use BCEWithLogits loss")
        else:
            loss_fn = torch.nn.CrossEntropyLoss().to(device)
            print("use CrossEntropy loss")

    if cfg_train["optimizer"]["type"] == "SGD":
        optimizer = optim.SGD(model.parameters(), 
                            lr=cfg_train["optimizer"]["kwargs"]["lr"],
                            momentum=cfg_train["optimizer"]["kwargs"]["momentum"], 
                            weight_decay=cfg_train["optimizer"]["kwargs"]["weight_decay"])
        print("Use SGD optimizer!!!")
    else:
        optimizer = optim.Adam(model.parameters(),
                            lr=cfg_train["optimizer"]["kwargs"]["lr"],
                            weight_decay=cfg_train["optimizer"]["kwargs"]["weight_decay"],
                            betas=(cfg_train["optimizer"]["kwargs"]["betas_0"], cfg_train["optimizer"]["kwargs"]["betas_1"]))
        print("Use Adam optimizer!!!")
    if cfg_train["lr_scheduler"]["mode"] == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, cfg_train["lr_scheduler"]["kwargs"]["step_size"],
                                                gamma=cfg_train["lr_scheduler"]["kwargs"]["gamma"])
    elif cfg_train["lr_scheduler"]["mode"] == "cosine":
        # warm_up_with_cosine_lr
        warm_up_with_cosine_lr = lambda epoch: epoch / cfg_train["lr_scheduler"]["warm_up_epochs"] if epoch <= cfg_train["lr_scheduler"]["warm_up_epochs"] else 0.5 * (math.cos((epoch - cfg_train["lr_scheduler"]["warm_up_epochs"]) /(cfg_train["epoch"] - cfg_train["lr_scheduler"]["warm_up_epochs"]) * math.pi) + 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    if cfg_train['resume'] != '':
        optimizer.load_state_dict(model_tmp["optimizer"])
    summary_train = {
        'loss': float('inf'), 'epoch': 0, 'step': 0
    }
    summary_valid = {
        'loss': float('inf'), 'epoch': 0, 'step': 0, 'acc': 0
    }
    summary_test = {
        'loss': float('inf'), 'epoch': 0, 'step': 0, 'acc': 0
    }
    summary_writer = SummaryWriter(log_save_path)
    Precision_flag = 0
    F1_flag = 0
    acc_valid_end=[]
    acc_test_end=[]

    is_test = False
    # summary(model,(3, 9, 224, 224))
    print(model)

    train_dataset = ImageFolderDataset(
        root=os.path.join(cfg_dataset['data_root'], cfg_dataset['train']['patch_cls_path']),
        img_size=cfg_dataset['image_size'],
        patch_size=cfg_dataset['patch_size'],
        crop_size=cfg_dataset['crop_size'],
        way="train",
        stain_normalizer=False,
        cfg=cfg,
        use_levels=cfg_dataset.get("use_levels", [0])
    )

    val_dataset = ImageFolderDataset(
        root=os.path.join(cfg_dataset['data_root'], cfg_dataset['val']['patch_cls_path']),
        img_size=cfg_dataset['image_size'],
        patch_size=cfg_dataset['patch_size'],
        crop_size=cfg_dataset['crop_size'],
        way="val",
        stain_normalizer=False,
        cfg=cfg,
        use_levels=cfg_dataset.get("use_levels", [0])
    )

    test_dataset = ImageFolderDataset(
        root=os.path.join(cfg_dataset['data_root'], cfg_dataset['test']['patch_cls_path']),
        img_size=cfg_dataset['image_size'],
        patch_size=cfg_dataset['patch_size'],
        crop_size=cfg_dataset['crop_size'],
        way="val",
        stain_normalizer=False,
        cfg=cfg,
        use_levels=cfg_dataset.get("use_levels", [0])
    )

    dataloader_train = DataLoader(train_dataset,
                                  batch_size=batch_size_train,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  shuffle=True)
    dataloader_val = DataLoader(val_dataset,
                                 batch_size=batch_size_valid,
                                 num_workers=num_workers,
                                 drop_last=False,
                                 shuffle=False)

    dataloader_test = DataLoader(test_dataset,
                                  batch_size=batch_size_valid,
                                  num_workers=num_workers,
                                  drop_last=False,
                                  shuffle=False)
    print("len(test_dataset):", len(test_dataset))


    for epoch in range(cfg_train["start_epoch"], cfg_train["epoch"]):

        confusion_train = ConfusionMatrix(num_classes=cfg["model"]["num_classes"],
                                          labels_dic=train_dataset.class_to_idx,
                                          steps=len(dataloader_train),
                                          grid_size=grid_size,
                                          batch_size=batch_size_train,
                                          datset_len=len(train_dataset),
                                          cfg=cfg,
                                          way="train")


        summary_train = train_epoch(summary_train, summary_writer, epoch, model,
                                    loss_fn, optimizer, dataloader_train, confusion_train)
        lr_scheduler.step()
        torch.save({'epoch': summary_train['epoch'],
                    'step': summary_train['step'],
                    'state_dict': model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   (ckpt_save_path + '/' + f'{epoch}.ckpt'))
        print(cfg_valid["log_every"])
        print(f"=========================epoch {epoch}:log=================================")

        confusion_valid = ConfusionMatrix(num_classes=cfg["model"]["num_classes"],
                                        labels_dic=val_dataset.class_to_idx,
                                        steps=len(dataloader_val),
                                        batch_size=batch_size_valid,
                                        grid_size=grid_size,
                                        datset_len = len(val_dataset),
                                        cfg=cfg,
                                        way="val")
        print("start to validate ........")
        summary_valid = valid_epoch(summary_valid, summary_writer, epoch, model, loss_fn,
                                    dataloader_val, confusion_valid)

        acc_valid_end.append(summary_valid['acc'])

        print("start to test ........")

        confusion_test = ConfusionMatrix(num_classes=cfg["model"]["num_classes"],
                                          labels_dic=test_dataset.class_to_idx,
                                          steps=len(dataloader_test),
                                          batch_size=batch_size_valid,
                                          grid_size=grid_size,
                                          datset_len=len(test_dataset),
                                          cfg=cfg,
                                          way="test")
        print("Test acc:")
        summary_test = valid_epoch(summary_test, summary_writer, epoch, model, loss_fn,
                                    dataloader_test, confusion_test)

        acc_test_end.append(summary_test['acc'])
        print(acc_test_end)

    summary_writer.close()


if __name__ == '__main__':
    ### initial
    global args, cfg, log_save_path, ckpt_save_path
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device("cuda")
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--cfg_path', '-c',
                        default='./work_dirs/1010_res34_1cls_v1.0.0/crf_config_F.yaml',
                        metavar='CFG_PATH',
                        type=str,
                        help='Path to the config file in yaml format')
    args = parser.parse_args()
    cfg = yaml.load(open(args.cfg_path, 'r'), Loader=yaml.Loader)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["train"]["device_ids"]
    log_save_path = os.path.join(cfg["saver"]["log_save_path"], cfg["saver"]["experiment_name"])
    ckpt_save_path = os.path.join(cfg["saver"]["ckpt_save_path"], cfg["saver"]["experiment_name"])
    os.makedirs(ckpt_save_path, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    print(cfg)
    print("torch.cuda.is_available(): ", torch.cuda.is_available(), 
        "\nlog_save_path:", log_save_path, 
        "\nckpt_save_path:", ckpt_save_path)
    run()
