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
import os
import sys
import time
import yaml
import math

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import cv2
# from torch import optim
from torch.nn import DataParallel, functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
torch.set_printoptions(threshold=np.inf)

sys.path.append('../../..')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


from Classification.model import MODELS
from Classification.dataset.wsi_producer_new import GridWSIPatchDataset

def valid_epoch(model, dataloader_valid):
    model.eval()
    steps = len(dataloader_valid)
    batch_size = dataloader_valid.batch_size
    grid_size = dataloader_valid.dataset._grid_size
    dataiter = iter(dataloader_valid)
    time_now = time.time()

    res_file = open(os.path.join(cfg["saver"]["ckpt_save_path"], cfg["saver"]["experiment_name"], cfg["saver"]["test_result"]), "a", encoding="utf-8")

    cancer_nums, use_sort_cancer_nums = 0, 0

    waiting_list_source_imgs_info = []
    waiting_list_logits = []


    with torch.no_grad():
        for step in range(steps):
            time_now = time.time()
            data, tif_name, x, y, source_img_flat, tif_path, region_img= next(dataiter)
            data = data.to(device)
            b, g, c, h, w = source_img_flat.shape
            output,  _= model(data) 
            if cfg["model"]["num_classes"] == 1:
                output = output.contiguous().view(-1)
            else:
                output = output.view(len(tif_name), -1)
            time_spent = time.time() - time_now
            logging.info(
            '{}, tif_name : {}, Step : {}, Run Time : {:.2f}'
                .format(time.strftime("%Y-%m-%d %H:%M:%S"), tif_name[0], step, time_spent))

            if cfg["model"]["num_classes"] == 1:
                output = output.sigmoid()
                preds = torch.zeros(b * g)
                preds[output > cfg['test']['thresh']] = 1
            else:
                output = F.softmax(output, dim=1)
                preds_max_logits, preds = torch.max(output, dim=1)

            infer_res_vis_path = os.path.join(args.infer_save_dir, cfg["saver"]["experiment_name"], tif_path[0].split("/")[-2],tif_name[0])

            os.makedirs(infer_res_vis_path, exist_ok=True)
            grid_pixels = cfg["dataset"]["patch_size"]*cfg["dataset"]["patch_size"]
            grid_white_thresh = grid_pixels*255*0.5

            # print('**'*40, preds.shape, len(tif_name), x.shape, y.shape, source_img_flat.shape)
            # torch.Size([540]) 60 torch.Size([60, 9, 3, 256, 256])
            
            if preds.sum() >= 0: # 判断batch里是否有肿瘤
                source_img_flat = source_img_flat.contiguous().view(-1, c, h, w)
                x = x.contiguous().view(-1)
                y = y.contiguous().view(-1)
                for idx in range(preds.shape[0]): # 循环grid
                    if cfg["model"]["num_classes"] == 1: # 一分类的情况下
                        res_dic = {}
                        if int(preds[idx].item()) == 1: # 是肿瘤就保存信息到json
                            res_dic["filename"] = tif_name[0]
                            res_dic["grid_idx"] = idx % grid_size
                            res_dic["pred"] = int(preds[idx].item())
                            # idx_= math.floor(idx/len(x))
                            res_dic["x"], res_dic["y"] = x[idx].item(), y[idx].item()
                            res_dic["pred_logit"] = round(output[idx].item(), 4)
                            # CHW,RGB-->HWC,RGB
                            grid_img = source_img_flat[idx].cpu().numpy().transpose((1, 2, 0))
                            # CHW,RGB-->CHW,BGR, denormal
                            grid_img = grid_img[:, :, ::-1]*128 + 128
                            grid_img_Gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
                            # cv2.imwrite(os.path.join(infer_res_vis_path,
                            #             tif_name[0]+"_"+str(x[idx].item())+"_"+str(y[idx].item())+str(res_dic["pred_logit"])+".png"), 
                            #             grid_img)
                            # if np.sum(grid_img_Gray) < grid_white_thresh:
                            if len(np.where(grid_img_Gray>210)[0])/grid_pixels < 0.12:
                                waiting_list_source_imgs_info.append([tif_name[0], res_dic["x"], res_dic["y"], round(output[idx].item(), 4), grid_img])
                                waiting_list_logits.append(output[idx].item())
                                use_sort_cancer_nums += 1
                            cancer_nums += 1
                        else:
                            continue
                        
                    else:
                        res_dic["pred_logit"] = round(preds_max_logits[idx], 4)

                    res_file.write(json.dumps(res_dic)+"\n")
            # if saved_nums >= cfg["dataset"]["valid"]["save_number"]:
            #     break
        print("predict cancer numbers: {}, use_sort_cancer_nums: {}".format(cancer_nums, use_sort_cancer_nums))
        # print("--------------->", len(waiting_list_source_imgs_info), np.array(waiting_list_source_imgs_info).shape)
        # print("--------------->", len(waiting_list_logits), waiting_list_logits)
        sort_index = sorted(range(len(waiting_list_logits)), key=lambda k: waiting_list_logits[k], reverse=True)
        save_nums = 0
        for i in range(min(use_sort_cancer_nums, args.save_patch_num)):
            cv2.imwrite(os.path.join(infer_res_vis_path,
                        waiting_list_source_imgs_info[sort_index[i]][0]+"_" \
                        +str(waiting_list_source_imgs_info[sort_index[i]][1])+"_" \
                        +str(waiting_list_source_imgs_info[sort_index[i]][2])+"_" \
                        +str(waiting_list_source_imgs_info[sort_index[i]][3])+".png"),
                        waiting_list_source_imgs_info[sort_index[i]][4])
            save_nums += 1

        print(tif_name[0], " saved_nums=======>>", save_nums)
    
    return 1

def run():
    logging.basicConfig(level=logging.INFO)
    cfg_dataset, cfg_test, cfg_model = cfg["dataset"], cfg["test"],  cfg["model"]
    num_GPU = 1
    batch_size_valid = cfg_test['batch_size_perGPU'] * num_GPU
    num_workers = cfg_test['num_workers_perGPU'] * num_GPU

    image_size, patch_size, crop_size = cfg['dataset']['image_size'], cfg['dataset']['patch_size'], cfg['dataset']['crop_size']
    if image_size % patch_size != 0:
        raise Exception('Image size / patch size != 0 : {} / {}'.
                        format(image_size, patch_size))
    patch_per_side = image_size // patch_size
    grid_size = patch_per_side * patch_per_side
    
    model = MODELS[cfg_model['model_name']](num_classes=cfg_model["num_classes"], num_nodes=grid_size, use_crf=cfg_model['use_crf'], pretrained=True)
    print("Use ckpt: ", cfg_test['weights_path'])
    model_file = torch.load(cfg_test['weights_path'])
    model = DataParallel(model, device_ids=None)
    model.load_state_dict(model_file['state_dict'])
    model = model.to(device)

    error_tif = []
    ##############################################

    infer_data_list = []
    suffix_name = []
    for idata in os.listdir(args.infer_dir):
        if "YouAn" in args.infer_dir:
            infer_data_list.append(idata)
            suffix_name.append(".mrxs")
        else:
            if idata.endswith(".tif"):
                infer_data_list.append(idata.split(".tif")[0])
                suffix_name.append(".tif")
            elif idata.endswith(".svs"):
                infer_data_list.append(idata.split(".svs")[0])
                suffix_name.append(".svs")
    ##############################################


    # print(cfg_dataset["valid"]["tif_names"])
    # for name in tqdm(cfg_dataset["valid"]["tif_names"]):
    for name, _suffix in tqdm(zip(infer_data_list, suffix_name)):
        if "YouAn" in args.infer_dir:
            sub_name = name.split("+")[1]
            tif_path = os.path.join(args.infer_dir, name, sub_name + _suffix)
        else:
            tif_path = os.path.join(args.infer_dir, name + _suffix)
        # try:
        dataset_valid = GridWSIPatchDataset(tif_path, name,
                                        image_size=cfg_dataset['image_size'],
                                        patch_size=cfg_dataset['patch_size'],
                                        crop_size=cfg_dataset['crop_size'])

        dataloader_valid = DataLoader(dataset_valid,
                                    batch_size=batch_size_valid,
                                    num_workers=num_workers,
                                    drop_last=False,
                                    shuffle=False)
        print("len(dataset_valid):", len(dataset_valid))
        print("start to validate ........")
        valid_epoch(model, dataloader_valid)
        # except:
        #     print('error is ',tif_path)
        #     error_tif.append(tif_path)
        
    print("Error tif =====> ", error_tif)



if __name__ == '__main__':
    ### initial
    global args, cfg
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device("cuda")
    parser = argparse.ArgumentParser(description='Infer model')
    parser.add_argument('--cfg_path', '-c',
                        default='./work_dirs/1102_res34_69tif_1cls_v2.0.0/config.yaml',
                        metavar='CFG_PATH',
                        type=str,
                        help='Path to the config file in yaml format')
    parser.add_argument('--infer_dir', '-i',
                        default='/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/prognostic/huashan',
                        type=str,
                        help='Path to the WSI data')
    parser.add_argument('--infer_save_dir', '-s',
                        default='/media/whisper/Newsmy/WSI_PatchData/CY_SL_FD_HS',
                        type=str,
                        help='Path to the WSI data')
    parser.add_argument('--save_patch_num', '-n',
                        default=20,
                        type=int,
                        help='The number of patch to save for one WSI')
    args = parser.parse_args()
    cfg = yaml.load(open(args.cfg_path, 'r'), Loader=yaml.Loader)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(cfg)
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    run()
