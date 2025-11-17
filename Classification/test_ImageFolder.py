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
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
# from torch import optim
from torch.nn import DataParallel, functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.append('.')

from dataset.Colon_loader import ImageFolderDataset

# from torchsummary import summary
# from openslide import OpenSlide
torch.set_printoptions(threshold=np.inf)

sys.path.append('../../..')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


from model import MODELS
from tools.utils import ConfusionMatrix, load_tif_dict, load_pretrain_checkpoint

def valid_epoch(epoch, model, dataloader_valid, confusion):
    model.eval()
    steps = len(dataloader_valid)
    dataiter = iter(dataloader_valid)
    grid_size = (cfg["dataset"]["image_size"]//cfg["dataset"]["patch_size"])**2

    res_file = open(os.path.join(cfg["saver"]["ckpt_save_path"], cfg["saver"]["experiment_name"], cfg["saver"]["test_result"]), "w", encoding="utf-8")

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
            else:
                output = output.view(len(target), -1)
            confusion.update(output.clone().detach().cpu(), target.clone().detach().cpu().numpy())
            time_spent = time.time() - time_now
            logging.info('{}, Epoch : {}, Run Time : {:.2f}'.format(time.strftime("%Y-%m-%d %H:%M:%S"), epoch, time_spent))
            # if cfg["model"]["num_classes"] == 1:
            #     output = output.sigmoid()
            #     preds = torch.zeros_like(target)
            #     preds[output > cfg['val']['thresh']] = 1
            # else:
            #     output = F.softmax(output, dim=1)
            #     preds_max_logits, preds = torch.max(output, dim=1)
            # for idx in range(len(target)):
            #     res_dic = {}
            #     res_dic["filename"] = img_name[idx//grid_size]
            #     name_split = res_dic["filename"].split("_")
            #     x_center, y_center = int(name_split[-2]), int(name_split[-1].strip(".png"))
            #     res_dic["x_center"], res_dic["y_center"] = x_center, y_center
            #     res_dic["grid_idx"] = idx % grid_size
            #     res_dic["is_correct"] = (target[idx].item()==preds[idx].item())
            #     res_dic["label"] = int(target[idx].item())
            #     res_dic["pred"] = int(preds[idx].item())
            #     if cfg["model"]["num_classes"] == 1:
            #         res_dic["pred_logit"] = round(output[idx].item(), 4)
            #     else:
            #         res_dic["pred_logit"] = round(preds_max_logits[idx].item(), 4)
            #
            #     if cfg['saver'].get('feats', False):
            #         res_dic["feats"] = output_tuple[1][idx//grid_size][idx%grid_size].clone().detach().cpu().numpy().tolist()
            #
            #     res_file.write(json.dumps(res_dic)+"\n")

    confusion.summary()

def run():
    logging.basicConfig(level=logging.INFO)
    cfg_dataset, cfg_test, cfg_saver, cfg_model = cfg["dataset"], cfg["test"], cfg["saver"], cfg["model"]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_GPU = 1
    batch_size_valid = cfg_test['batch_size_perGPU'] * num_GPU
    num_workers = cfg_test['num_workers_perGPU'] * num_GPU

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

    print("Use ckpt: ", cfg_test['weights_path'])
    model_file = torch.load(cfg_test['weights_path'])
    model.load_state_dict(model_file['state_dict'])
    model = model.to(device)

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

    dataloader_test = DataLoader(test_dataset,
                                 batch_size=batch_size_valid,
                                 num_workers=num_workers,
                                 drop_last=False,
                                 shuffle=False)
    print("len(dataset_test):", len(test_dataset))

    confusion_test = ConfusionMatrix(num_classes=cfg["model"]["num_classes"],
                                    labels_dic=test_dataset.class_to_idx,
                                    steps=len(dataloader_test),
                                    batch_size=batch_size_valid,
                                    grid_size=grid_size, 
                                    datset_len = len(test_dataset),
                                    cfg=cfg,
                                    way="test")
    print("start to test ........")
    epoch = 0
    summary_valid = valid_epoch(epoch, model, dataloader_test, confusion_test)

if __name__ == '__main__':
    ### initial
    global args, cfg, log_save_path, ckpt_save_path
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device("cuda")
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('--cfg_path', '-c',
                        # default='./work_dirs/DDSNet/config.yaml',
                        default='./work_dirs/LC25000Colon/config.yaml',
                        # default='./work_dirs/LC25000Lung/config.yaml',
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
        "\nlog_save_path:", log_save_path)
    run()
