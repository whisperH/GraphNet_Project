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

sys.path.append('../..')
sys.path.append('/')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from model import MODELS
from dataset.wsi_producer_new import GridWSIPatchDataset

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

patch_list = []
for ineed_do in ['D18-02242-04', "D18-02107-10"]:
    for ipatch in os.listdir(
            os.path.join("/media/server/Newsmy/WSI_PatchData/CY_SL_FD_HS/250715_res34_upload/FullAnnotation_96DPI",
                         ineed_do)):
        if "CAM" in ipatch:
            continue
        HE_name, x, y, conf = ipatch.split("_")
        patch_list.append(f"{HE_name}_{x}_{y}")


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        # cancer_prob = model_output.sigmoid()
        return model_output.squeeze().sum()


class myCAM(GradCAM):
    def get_target_width_height(self, input_tensor: torch.Tensor):
        if len(input_tensor.shape) == 4:
            width, height = 256, 256
            return width, height
        elif len(input_tensor.shape) == 5:
            depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
            return depth, width, height
        else:
            raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")

    def forward(
            self, input_tensor, targets, eigen_smooth=False
    ):
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        # if targets is None:
        #     target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        #     targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            all_output = []
            for output in outputs:
                all_output.append(targets[0](output))
            loss = sum(all_output)
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                # keep the computational graph, create_graph = True is needed for hvp
                torch.autograd.grad(loss, input_tensor, retain_graph=True, create_graph=True)
                # When using the following loss.backward() method, a warning is raised: "UserWarning: Using backward() with create_graph=True will create a reference cycle"
                # loss.backward(retain_graph=True, create_graph=True)
            if 'hpu' in str(self.device):
                self.__htcore.mark_step()

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        b, _, g, c, h, w = input_tensor.shape  # center crop size
        cam_per_layer = self.compute_cam_per_layer(input_tensor.contiguous().view(-1, c, h, w), targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)


def valid_epoch(model, dataloader_valid):
    model.eval()
    steps = len(dataloader_valid)
    batch_size = dataloader_valid.batch_size
    grid_size = dataloader_valid.dataset._grid_size
    dataiter = iter(dataloader_valid)
    time_now = time.time()

    res_file = open(
        os.path.join(cfg["saver"]["ckpt_save_path"], cfg["saver"]["experiment_name"], cfg["saver"]["test_result"]), "a",
        encoding="utf-8")

    cancer_nums, use_sort_cancer_nums = 0, 0

    waiting_list_source_imgs_info = []
    waiting_list_logits = []

    target_layers = [model.module.layer4[-1]]
    targets = [ClassifierOutputTarget(1)]

    for step in range(steps):
        time_now = time.time()
        data, tif_name, x, y, source_img_flat, tif_path, region_img = next(dataiter)
        # print(f"processing {tif_name}")
        data = data.to(device)
        b, g, c, h, w = source_img_flat.shape
        ori_output = model(data)
        if cfg["model"]["num_classes"] == 1:
            output = ori_output.contiguous().view(-1)
        else:
            output = ori_output.view(len(tif_name), -1)

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

        if "YouAn" in tif_path[0]:
            infer_res_vis_path = os.path.join(args.infer_save_dir, cfg["saver"]["experiment_name"],
                                              tif_path[0].split("/")[-3], tif_name[0])
        else:
            infer_res_vis_path = os.path.join(args.infer_save_dir, cfg["saver"]["experiment_name"],
                                              tif_path[0].split("/")[-2], tif_name[0])

        os.makedirs(infer_res_vis_path, exist_ok=True)
        grid_pixels = cfg["dataset"]["patch_size"] * cfg["dataset"]["patch_size"]

        # print('**'*40, preds.shape, len(tif_name), x.shape, y.shape, source_img_flat.shape)
        # torch.Size([540]) 60 torch.Size([60, 9, 3, 256, 256])
        # Construct the CAM object once, and then re-use it on many images.

        if preds.sum() >= 0:  # 判断batch里是否有肿瘤
            with myCAM(model=model, target_layers=target_layers) as cam:
                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input_tensor=data, targets=targets)
                # In this example grayscale_cam has only one image in the batch:
                # grayscale_cam = grayscale_cam[0, :]
                # You can also get the model outputs without having to redo inference
                # model_outputs = cam.outputs

            source_img_flat = source_img_flat.contiguous().view(-1, c, h, w)
            x = x.contiguous().view(-1)
            y = y.contiguous().view(-1)
            for idx in range(preds.shape[0]):  # 循环grid
                if cfg["model"]["num_classes"] == 1:  # 一分类的情况下
                    res_dic = {}
                    if int(preds[idx].item()) == 1:  # 是肿瘤就保存信息到json
                        res_dic["filename"] = tif_name[0]
                        res_dic["grid_idx"] = idx % grid_size
                        res_dic["pred"] = int(preds[idx].item())
                        # idx_= math.floor(idx/len(x))
                        res_dic["x"], res_dic["y"] = x[idx].item(), y[idx].item()
                        res_dic["pred_logit"] = round(output[idx].item(), 4)
                        # CHW,RGB-->HWC,RGB
                        grid_img = source_img_flat[idx].cpu().numpy().transpose((1, 2, 0))
                        # CHW,RGB-->CHW,BGR, denormal
                        grid_img = grid_img[:, :, ::-1] * 128 + 128
                        grid_img_Gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)

                        visualization = show_cam_on_image(grid_img.astype(np.float32) / 255.0, grayscale_cam[idx],
                                                          use_rgb=True)
                        # if f'{tif_name[0]}_{res_dic["x"]}_{res_dic["y"]}' not in patch_list:
                        #     continue
                        # cv2.imwrite(os.path.join(infer_res_vis_path,
                        #             tif_name[0]+"_"+str(x[idx].item())+"_"+str(y[idx].item())+str(res_dic["pred_logit"])+".png"),
                        #             grid_img)
                        # if np.sum(grid_img_Gray) < grid_white_thresh:
                        if len(np.where(grid_img_Gray > 210)[0]) / grid_pixels < 0.5:
                            waiting_list_source_imgs_info.append(
                                [
                                    tif_name[0],
                                    res_dic["x"],
                                    res_dic["y"],
                                    round(output[idx].item(), 4),
                                    grid_img,
                                    visualization
                                ]
                            )
                            waiting_list_logits.append(output[idx].item())
                            use_sort_cancer_nums += 1
                        cancer_nums += 1
                    else:
                        continue

                else:
                    res_dic["pred_logit"] = round(preds_max_logits[idx], 4)

                res_file.write(json.dumps(res_dic) + "\n")
        # if use_sort_cancer_nums >= cfg["dataset"]["valid"]["save_number"] * 50:
        #     break
    print("predict cancer numbers: {}, use_sort_cancer_nums: {}".format(cancer_nums, use_sort_cancer_nums))
    # print("--------------->", len(waiting_list_source_imgs_info), np.array(waiting_list_source_imgs_info).shape)
    # print("--------------->", len(waiting_list_logits), waiting_list_logits)
    sort_index = sorted(range(len(waiting_list_logits)), key=lambda k: waiting_list_logits[k], reverse=True)
    save_nums = 0
    for i in range(min(use_sort_cancer_nums, args.save_patch_num)):
        cv2.imwrite(os.path.join(infer_res_vis_path,
                                 waiting_list_source_imgs_info[sort_index[i]][0] + "_" \
                                 + str(waiting_list_source_imgs_info[sort_index[i]][1]) + "_" \
                                 + str(waiting_list_source_imgs_info[sort_index[i]][2]) + "_" \
                                 + str(waiting_list_source_imgs_info[sort_index[i]][3]) + ".png"),
                    waiting_list_source_imgs_info[sort_index[i]][4])

        cv2.imwrite(os.path.join(infer_res_vis_path,
                                 waiting_list_source_imgs_info[sort_index[i]][0] + "_" \
                                 + str(waiting_list_source_imgs_info[sort_index[i]][1]) + "_" \
                                 + str(waiting_list_source_imgs_info[sort_index[i]][2]) + "_" \
                                 + str(waiting_list_source_imgs_info[sort_index[i]][3]) + "_CAM.png"),
                    waiting_list_source_imgs_info[sort_index[i]][5])
        save_nums += 1

    print(tif_name[0], " saved_nums=======>>", save_nums)

    return 1


def run():
    logging.basicConfig(level=logging.INFO)
    cfg_dataset, cfg_test, cfg_model = cfg["dataset"], cfg["test"], cfg["model"]
    num_GPU = len(args.device_id.split(","))
    batch_size_valid = 1
    num_workers = cfg_test['num_workers_perGPU'] * num_GPU

    image_size, patch_size, crop_size = cfg['dataset']['image_size'], cfg['dataset']['patch_size'], cfg['dataset'][
        'crop_size']
    if image_size % patch_size != 0:
        raise Exception('Image size / patch size != 0 : {} / {}'.
                        format(image_size, patch_size))
    patch_per_side = image_size // patch_size
    grid_size = patch_per_side * patch_per_side

    model = MODELS[cfg_model['model_name']](
        num_classes=cfg_model["num_classes"],
        num_nodes=grid_size,
        use_crf=cfg_model['use_crf'],
        pretrained=True,
        use_levels=cfg_dataset.get("use_levels", [0]),
        level_crf=cfg_model.get("level_crf", False),
        concat_level_feats=cfg_model.get("concat_level_feats", False),
        cfg=cfg
    )
    print("Use ckpt: ", cfg_test['weights_path'])
    model_file = torch.load(cfg_test['weights_path'])
    model = DataParallel(model, device_ids=[_ for _ in range(num_GPU)])
    model.load_state_dict(model_file['state_dict'])

    model = model.to(device)

    error_tif = []
    ##############################################
    center_name = args.infer_dir.split("/")[-1]
    check_res_vis_path = os.path.join(args.infer_save_dir, cfg["saver"]["experiment_name"], center_name)
    os.makedirs(check_res_vis_path, exist_ok=True)
    done_list = []
    for idone in []:
        if len(os.listdir(os.path.join(check_res_vis_path, idone))) >= args.save_patch_num:
            done_list.append(idone)

    infer_data_list = []
    suffix_name = []
    for idata in os.listdir(args.infer_dir):
        if "ZY32_2_Four26656_26657_ZHL" in args.infer_dir:
            if (idata not in done_list) and (idata.endswith(".mrxs")):
                infer_data_list.append(idata.split(".mrxs")[0])
                suffix_name.append(".mrxs")
            else:
                print(f"{idata} has been done!")
        else:
            if idata.endswith(".tif"):
                sample_name = idata.split(".tif")[0]
                if sample_name not in done_list:
                    infer_data_list.append(sample_name)
                    suffix_name.append(".tif")
                else:
                    print(f"{sample_name} has been done!")
            elif idata.endswith(".svs"):
                sample_name = idata.split(".svs")[0]
                if sample_name not in done_list:
                    infer_data_list.append(sample_name)
                    suffix_name.append(".svs")
                else:
                    print(f"{sample_name} has been done!")

    print(f"The length of infer list {len(infer_data_list)}")

    for name, _suffix in tqdm(zip(infer_data_list, suffix_name)):
        if "YouAn" in args.infer_dir:
            sub_name = name.split("+")[1]
            tif_path = os.path.join(args.infer_dir, name, sub_name + _suffix)
        else:
            tif_path = os.path.join(args.infer_dir, name + _suffix)
        # try:
        print(f"tif_path:{tif_path}")
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

    print("Error tif =====> ", error_tif)


if __name__ == '__main__':
    ### initial
    global args, cfg
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device("cuda:0")
    parser = argparse.ArgumentParser(description='Infer model')
    parser.add_argument('--cfg_path', '-c',
                        default='./work_dirs/1102_res34_69tif_1cls_v2.0.0/config.yaml',
                        metavar='CFG_PATH',
                        type=str,
                        help='Path to the config file in yaml format')
    parser.add_argument('--infer_dir', '-i',
                        default='/media/server/Newsmy/WSI_RawData/CY_SL_FD_HS/MoleculeValidation/FullAnnotation',
                        type=str,
                        help='Path to the WSI data')
    parser.add_argument('--infer_save_dir', '-s',
                        default='/media/server/Newsmy/WSI_PatchData/CY_SL_FD_HS',
                        type=str,
                        help='Path to the WSI data')
    parser.add_argument('--save_patch_num', '-n',
                        default=20,
                        type=int,
                        help='The number of patch to save for one WSI')
    parser.add_argument('--device_id', '-d',
                        default="2",
                        type=str,
                        help='The number of patch to save for one WSI')
    args = parser.parse_args()
    cfg = yaml.load(open(args.cfg_path, 'r'), Loader=yaml.Loader)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    print(cfg)
    print("torch.cuda.is_available(): ", torch.cuda.is_available())
    run()
