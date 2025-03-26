# -*-coding:utf-8-*-
import os
import sys
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from PIL import ImageFile
import random
from os.path import getsize
from openslide import OpenSlide
import cv2
import json
from tqdm import tqdm
import logging
# import pyvips

import staintools
import tifffile



class WsiDataset(Dataset):
    def __init__(self, json_path, img_size, patch_size,
                crop_size=224, normalize=True, way="train", key_word="",
                sample_class_num={}, wsi_lst=None, is_test=False,
                stain_normalizer=False, cfg=None, level=0, total_tifs=None,
                total_labels=None, use_levels=[0]):
        self._json_path = json_path
        self._img_size = img_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        # self._key_word = key_word
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        self._way = way
        self.wsi_lst = wsi_lst
        # self.sample_class_num = sample_class_num
        # self.is_test = is_test
        # self.valid_number_ratio = valid_number_ratio
        # self.stain_normalizer = stain_normalizer
        self.cfg = cfg
        self._level = level
        self.total_tifs = total_tifs
        self.total_labels = total_labels
        # self._annotations = {}
        self.use_levels = use_levels
        self._preprocess()
        self.transform = transforms.Compose([
            transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation((5)),
            # transforms.RandomCrop((self._img_size, self._img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.tmp_labels = {}

    def _preprocess(self):
        # {name:, wsi_path:, x_center:, y_center:, class:, grid_size:, level: }
        if self._img_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._img_size, self._patch_size))
        self._patch_per_side = self._img_size // self._patch_size
        self._patch_nums = self._patch_per_side * self._patch_per_side

        data_file = open(self._json_path, "r", encoding="utf-8")
        data_lines = data_file.readlines()
        data_file.close()
        self._coords = []
        for line in data_lines:
            data_dict = json.loads(line)
            if data_dict["wsi_name"] in self.wsi_lst:
                self._coords.append(data_dict)
        if self.cfg["dataset"].get("debug", False):
            self._coords = random.sample(self._coords, k=int(0.04*len(self._coords)))
        
    def __len__(self):
        return len(self._coords)

    def vips2numpy(self, vi):
        format_to_dtype = {
                            'uchar': np.uint8,
                            'char': np.int8,
                            'ushort': np.uint16,
                            'short': np.int16,
                            'uint': np.uint32,
                            'int': np.int32,
                            'float': np.float32,
                            'double': np.float64,
                            'complex': np.complex64,
                            'dpcomplex': np.complex128,
                        }
        return np.ndarray(buffer=vi.write_to_memory(),
                        dtype=format_to_dtype[vi.format],
                        shape=[vi.height, vi.width, vi.bands])




    def __getitem__(self, idx):
        cur_dict = self._coords[idx]
        # {name:, wsi_path:, x_center:, y_center:, class:, grid_size:, level: }

        ###################### label from json ################
        # wsi_name, wsi_path, top_left_h, top_left_w, image_size, patch_size, label_of_patches = cur_dict["wsi_name"], cur_dict["wsi_path"], cur_dict["top_left_h"], cur_dict["top_left_w"], self._img_size, self._patch_size, cur_dict["label_of_patches"]

        # img_name = wsi_name + '_' + str(top_left_w) + '_' + str(top_left_h)

        # label_of_patches = np.array(label_of_patches)
        # label_grid = np.reshape(label_of_patches, (self._patch_per_side, self._patch_per_side))

        #################### online lable #########################

        wsi_name, wsi_path, top_left_h, top_left_w, image_size,patch_size = cur_dict["wsi_name"], cur_dict["wsi_path"], cur_dict["top_left_h"], cur_dict["top_left_w"], self._img_size, self._patch_size
        
        img_name = wsi_name + '_' + str(top_left_w) + '_' + str(top_left_h)

        # label_path = wsi_path[:-4] + "_viable.tif"
        # label_path = wsi_path[:-4] + "_mask.tif"

        ######## 0 ##########
        # label_tif = tifffile.imread(label_path) // 255

        ######## 1 ###########
        # label_tif = self.vips2numpy(pyvips.Image.new_from_file(label_path))[:, :, :3].squeeze() // 255

        ######## 2 ###########
        # if wsi_name in self.tmp_labels.keys():
        #     label_tif = self.tmp_labels[wsi_name]
        # else:
        #     print("----refresh tmp_labels dict----")
        #     self.tmp_labels = {}
        #     self.tmp_labels[wsi_name] = tifffile.imread(label_path) // 255
        #     label_tif = self.tmp_labels[wsi_name]


        ######## 3 ###########
        # label_tif = self.total_labels[wsi_name]
        # if label_tif.shape[0] != self.total_tifs[wsi_name].dimensions[1] or label_tif.shape[1] != self.total_tifs[wsi_name].dimensions[0]:
        #     print(label_path, "label shape: ",label_tif.shape, "wsi shape: ", self.total_tifs[wsi_name].dimensions)

        ########## 4 ###########
        mask_path = os.path.join("/".join(cur_dict["wsi_path"].split("/")[:-1]), "sub_image_mask", str(image_size), wsi_name + '_' + str(cur_dict["top_left_h"]) + '_' + str(cur_dict["top_left_w"]) + ".png")
        img_mask = cv2.imread(mask_path, 0)

        label_of_patches = np.zeros((self._patch_per_side, self._patch_per_side), dtype=np.float32)
        
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # patch_h = top_left_h + x_idx * patch_size
                # patch_w = top_left_w + y_idx * patch_size
                ## if the tumor pixels is bigger than non_tumor, patch label is tumor
                # if np.sum(label_tif[patch_h:patch_h+self._patch_size, patch_w:patch_w+self._patch_size]) >= 0.5*self._patch_size**2:
                #     label_of_patches[y_idx, x_idx] = 1
                # else:
                #     label_of_patches[y_idx, x_idx] = 0

                patch_h = x_idx * patch_size
                patch_w = y_idx * patch_size
                if np.sum(img_mask[patch_h:patch_h+patch_size, patch_w:patch_w+patch_size]) >= 0.5*self._patch_size**2:
                    label_of_patches[x_idx, y_idx] = 1
                else:
                    label_of_patches[x_idx, y_idx] = 0

        label_grid = label_of_patches

        # pixel_label_flat = label_tif[top_left_h:top_left_h+image_size, top_left_w:top_left_w+image_size].flatten()
        pixel_label_flat = img_mask.flatten()
        

        img_dic = {}
        label_dic = {}
        for level in self.use_levels:
            # x_range, y_range = self.total_tifs[pid].dimensions
            shift_w_top_left = max(0, top_left_w - (self._img_size*2**level - self._img_size)//2)
            shift_h_top_left = max(0, top_left_h - (self._img_size*2**level - self._img_size)//2)
            img = self.total_tifs[wsi_name].read_region(
                (shift_w_top_left, shift_h_top_left), level, (self._img_size, self._img_size)).convert('RGB')
            # color jitter
            if self._way == "train":
            #     img = self._color_jitter(img)
            #     # use left_right flip
            #     # if not self.cfg["model"].get("concat_level_feats", False):
            #     if np.random.rand() > 0.5:
            #         # img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #         label_grid = np.fliplr(label_grid)

            #     # use rotate，进行随机旋转，旋转90*num_rotate度
            #     num_rotate = np.random.randint(0, 4)
            #     img = img.rotate(90 * num_rotate)
            #     label_grid = np.rot90(label_grid, num_rotate)
            # # PIL image:   H x W x C
            # # torch image: C X H X W
            # img = self.transform(img)
            # img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
            # if self._normalize:
            #     img = (img - 128.0) / 128.0
                img = self.transform(img)
            elif self._way == "valid":
                img = self.transform_val(img)

            img_dic[level] = img

        if "resnet" in self.cfg["model"]["model_name"]:
            # x---->b,level,grid,c,h,w---->3,2,9,3,224,224
            img_flat_dic = {}
            idx = 0
            img_flat = np.zeros(
                (self._patch_nums, 3, self._crop_size, self._crop_size),
                dtype=np.float32)
                
            for k, v in img_dic.items():
                for x_idx in range(self._patch_per_side):
                    for y_idx in range(self._patch_per_side):
                        # center crop each patch
                        x_start = int(
                            (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                        x_end = x_start + self._crop_size
                        y_start = int(
                            (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                        y_end = y_start + self._crop_size
            
                        img_flat[idx] = v[:, x_start:x_end, y_start:y_end]
                        idx += 1

                img_flat_dic[level] = img_flat
                
            for idx,level in enumerate(self.use_levels):
                if idx == 0:
                    img_flat = np.expand_dims(img_flat_dic[level], 0)
                else:
                    img_flat = np.concatenate((img_flat, np.expand_dims(img_flat_dic[level], 0)), axis=0)
        else:
            # img: level, img_c, img_h, img_w
            for idx,level in enumerate(self.use_levels):
                if idx == 0:
                    img_flat = np.expand_dims(img_dic[level], 0)
                else:
                    img_flat = np.concatenate((img_flat, np.expand_dims(img_dic[level], 0)), axis=0)

        # label_flat: patch_size
        label_flat = label_of_patches.flatten()
        # print(img_name, img_flat.shape, label_flat.shape, pixel_label_flat.shape)
        return (img_flat, label_flat, pixel_label_flat, wsi_name, img_name, wsi_path)

## debug
if __name__=="__main__":

        from torch.utils.data import DataLoader
        import time
        from random import shuffle
        from sklearn.model_selection import KFold

        wsi_lst = ['01_01_0083', '01_01_0085', '01_01_0087', '01_01_0088', '01_01_0089', '01_01_0090', '01_01_0091', '01_01_0092', '01_01_0093', '01_01_0094', '01_01_0095', '01_01_0096', '01_01_0098', '01_01_0100', '01_01_0101', '01_01_0103', '01_01_0104', '01_01_0106', '01_01_0107', '01_01_0108', '01_01_0110', '01_01_0111', '01_01_0112', '01_01_0113', '01_01_0114', '01_01_0115', '01_01_0116', '01_01_0117', '01_01_0118', '01_01_0119', '01_01_0120', '01_01_0121', '01_01_0122', '01_01_0123', '01_01_0124', '01_01_0125', '01_01_0126', '01_01_0127', '01_01_0128', '01_01_0129', '01_01_0130', '01_01_0131', '01_01_0132', '01_01_0133', '01_01_0134', '01_01_0135', '01_01_0136', '01_01_0137', '01_01_0138', '01_01_0139']

        
        json_path = "/workspace/data1/huangxiaoshuang/json_files/paip2019_v3.json"

        tmp_lst = []
        with open(json_path, "r", encoding="utf-8") as data_file:
            data_lines = data_file.readlines()
            for line in data_lines:
                data_dict = json.loads(line)
                tmp_lst.append(data_dict)
        shuffle(tmp_lst)

        total_tifs = {}
        for data_dict in tqdm(tmp_lst):
        # for data_dict in tmp_lst:
            if data_dict["wsi_name"] not in total_tifs.keys():
                total_tifs[data_dict["wsi_name"]] = OpenSlide(data_dict["wsi_path"])
                # print(len(total_tifs.keys()))
            # if len(total_tifs.keys()) == 50:
            #     break
        print(total_tifs.keys(), len(total_tifs.keys()))

        ## 五折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index , test_index in kf.split(wsi_lst):  # 调用split方法切分数据

            import yaml
            cfg = yaml.load(open("/workspace/home/huangxiaoshuang/medicine/hcc-prognostic/Classification/WsiNet_work_dirs/WsiNet_v3.0.1_new/config.yaml", 'r'), Loader=yaml.Loader)
            
            train_wsi_lst = [wsi_lst[i] for i in train_index]
            test_wsi_lst = [wsi_lst[i] for i in test_index]
            # dataset_train = WsiDataset(json_path,
            #                                 768,
            #                                 128,
            #                                 128,
            #                                 total_tifs=total_tifs,
            #                                 wsi_lst=train_wsi_lst,
            #                                 cfg=cfg)
            dataset_test = WsiDataset(json_path,
                                            768,
                                            64,
                                            64,
                                            total_tifs=total_tifs,
                                            wsi_lst=test_wsi_lst,
                                            cfg=cfg)

            # print("len(dataset_train): ",len(dataset_train))
            print("len(dataset_test): ",len(dataset_test))

            # dataloader_train = DataLoader(dataset_train,
            #                                 batch_size=4,
            #                                 num_workers=1,
            #                                 drop_last=False,
            #                                 shuffle=True)
            dataloader_test = DataLoader(dataset_test,
                                            batch_size=200,
                                            num_workers=10,
                                            drop_last=False,
                                            shuffle=True)
            
            sys.path.append('../../..')

            from Classification.tools.utils import ConfusionMatrix  
            confusion = ConfusionMatrix(num_classes=cfg["model"]["num_classes"],
            labels_dic={"no_cancer": 0, "cancer": 1}, 
            steps=len(dataloader_test), 
            batch_size=200,
            grid_size=144, 
            datset_len = len(dataset_test),
            cfg=cfg,
            way="valid")
            import torch
            steps = len(dataloader_test)
            dataiter = iter(dataloader_test)
            for step in tqdm(range(steps)):
                time_now = time.time()
                
                datas, targets, pixel_label_flat, wsi_names, img_names, wsi_paths= next(dataiter)
                # print(targets.shape)
                b, n = targets.shape
                targets = targets.flatten()
                output = targets.clone().detach()
                # print(type(output), type(targets), type(pixel_label_flat))
                # print(output.shape, targets.shape, pixel_label_flat.shape)
                # break
                confusion.update(output, targets.clone().detach().cpu().numpy(), pixel_label_flat.detach().cpu().numpy())
            summery_dic = confusion.summary(vis=True)
            break
                
                