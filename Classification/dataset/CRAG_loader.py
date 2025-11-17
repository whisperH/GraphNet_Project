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
        self._coords = []
        for i in os.listdir(os.path.join(self._json_path, self._way, "Images")):
            if i.endswith(".png"):
                self._coords.append(i)

    def __len__(self):
        return len(self._coords)
    
    def __getitem__(self, idx):
        name = self._coords[idx][:-4]
        img_path = os.path.join(self._json_path,self._way, "Images", self._coords[idx])
        # img = Image.open(img_path)
        # img = img.resize((1536,1536),)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(1536,1536))
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
            img = self.transform(Image.fromarray(img.astype(np.uint8)))
        elif self._way == "valid":
            img = self.transform_val(Image.fromarray(img.astype(np.uint8)))
        
        # mask = Image.open(os.path.join(self._json_path, self._way, "Annotation",name+"_255.png"))//255
        # print("0",np.unique(np.array(mask)), np.array(mask).shape)
        # mask = mask.resize((1536,1536),)
        # print("1",np.unique(np.array(mask)), np.array(mask).shape)
        mask = cv2.imread(os.path.join(self._json_path, self._way, "Annotation",name+"_255.png"), 0)//255
        mask = cv2.resize(mask, (1536,1536))
        # print(np.unique(mask), mask.shape)
        patch_size = self._patch_size
        self._patch_per_side = 1536 // patch_size
        label_of_patches = np.zeros((self._patch_per_side * self._patch_per_side), dtype=np.float32)
        for idx in range(self._patch_per_side**2):
                patch_h = (idx//self._patch_per_side) * patch_size
                patch_w = (idx%self._patch_per_side) * patch_size
                if np.sum(mask[patch_h:patch_h+patch_size, patch_w:patch_w+patch_size]) >= 0.5*patch_size**2:
                    label_of_patches[idx] = 1
                else:
                    label_of_patches[idx] = 0
        # label_of_patches = np.zeros((self._patch_per_side, self._patch_per_side), dtype=np.float32)
        # for x_idx in range(self._patch_per_side):
        #     for y_idx in range(self._patch_per_side):
        #         patch_h = x_idx * patch_size
        #         patch_w = y_idx * patch_size
        #         if np.sum(mask[patch_h:patch_h+patch_size, patch_w:patch_w+patch_size]) >= 0.5*patch_size**2:
        #             label_of_patches[x_idx, y_idx] = 1
        #         else:
        #             label_of_patches[x_idx, y_idx] = 0
        pixel_label_flat = mask.flatten()
        img_dic = {}
        img_dic[0] = img
        for idx,level in enumerate(self.use_levels):
            if idx == 0:
                img_flat = np.expand_dims(img_dic[level], 0)
            else:
                img_flat = np.concatenate((img_flat, np.expand_dims(img_dic[level], 0)), axis=0)

        label_flat = label_of_patches.flatten()
        img_name = name +"_0_0"
        return (img_flat, label_flat, pixel_label_flat, name, img_name, img_path)


## debug
if __name__=="__main__":

    from torch.utils.data import DataLoader
    import yaml
    wsi_lst = ['01_01_0083', '01_01_0085', '01_01_0087', '01_01_0088', '01_01_0089', '01_01_0090', '01_01_0091', '01_01_0092', '01_01_0093', '01_01_0094', '01_01_0095', '01_01_0096', '01_01_0098', '01_01_0100', '01_01_0101', '01_01_0103', '01_01_0104', '01_01_0106', '01_01_0107', '01_01_0108', '01_01_0110', '01_01_0111', '01_01_0112', '01_01_0113', '01_01_0114', '01_01_0115', '01_01_0116', '01_01_0117', '01_01_0118', '01_01_0119', '01_01_0120', '01_01_0121', '01_01_0122', '01_01_0123', '01_01_0124', '01_01_0125', '01_01_0126', '01_01_0127', '01_01_0128', '01_01_0129', '01_01_0130', '01_01_0131', '01_01_0132', '01_01_0133', '01_01_0134', '01_01_0135', '01_01_0136', '01_01_0137', '01_01_0138', '01_01_0139']

    
    json_path = "/workspace/data1/medicine/CRAG_V2/"

    cfg = yaml.load(open("/workspace/home/huangxiaoshuang/medicine/hcc-prognostic/Classification/WsiNet_work_dirs/WsiNet_v3.0.1_new/config.yaml", 'r'), Loader=yaml.Loader)
    dataset_test = WsiDataset(json_path,
                                    768,
                                    64,
                                    64,
                                    total_tifs=None,
                                    wsi_lst=None,
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
    for i,data in enumerate(dataloader_test):
        pass