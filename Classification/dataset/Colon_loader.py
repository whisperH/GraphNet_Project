# -*-coding:utf-8-*-
import os
import sys
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from PIL import ImageFile
from PIL import Image

import cv2
import json
from tqdm import tqdm
import logging
from torchvision.datasets import ImageFolder

# import staintools
# import tifffile



class ImageFolderDataset(ImageFolder):
    def __init__(self, root, img_size, patch_size,
                crop_size=224, normalize=True, way="train", key_word="",
                wsi_lst=None, is_test=False,
                stain_normalizer=False, cfg=None, level=0, total_tifs=None,
                total_labels=None, use_levels=[0]):
        super().__init__(root)
        self._img_size = img_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        # self._key_word = key_word
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        self._way = way
        self.wsi_lst = wsi_lst
        # self.is_test = is_test
        # self.valid_number_ratio = valid_number_ratio
        # self.stain_normalizer = stain_normalizer
        self.cfg = cfg
        self._level = level
        self.total_tifs = total_tifs
        self.total_labels = total_labels
        self._preprocess()
        self.use_levels = use_levels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop((self._img_size, self._img_size)),
        ])

    def _preprocess(self):
        if self._img_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._img_size, self._patch_size))

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        name = self.imgs[idx]
        img_name = name[0].split("/")[-1]
        # the grid of labels for each patch
        label_grid_temp = np.zeros((self._patch_per_side, self._patch_per_side),
                                   dtype=np.float32)
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                if name[1] == 0:
                    label_grid_temp[y_idx, x_idx] = 0
                else:
                    label_grid_temp[y_idx, x_idx] = 1

        img_dic = {}
        label_dic = {}

        label_grid = np.copy(label_grid_temp)
        img = Image.open(name[0])
        # color jitter
        if self._way == "train":
            img = self._color_jitter(img)
            # use left_right flip
            if not self.cfg["model"].get("concat_level_feats", False):
                if np.random.rand() > 0.5:
                    # img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    label_grid = np.fliplr(label_grid)

                # use rotate，进行随机旋转，旋转90*num_rotate度
                num_rotate = np.random.randint(0, 4)
                img = img.rotate(90 * num_rotate)
                label_grid = np.rot90(label_grid, num_rotate)
        # # PIL image:   H x W x C
        # # torch image: C X H X W
        level = 0
        img = self.transform(img)
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
        if self._normalize:
            img = (img - 128.0) / 128.0
        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)
        img_flat_ = np.zeros(
            (self._grid_size, 3, 299, 299),
            dtype=np.float32)
        label_flat = np.zeros(self._grid_size, dtype=np.float32)
        idx = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int(
                    (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                y_end = y_start + self._crop_size
                img_flat[idx] = img[:, x_start:x_end, y_start:y_end]
                label_flat[idx] = label_grid[x_idx, y_idx]

                idx += 1
        img_dic[level] = img_flat
        label_dic[level] = label_flat

        for idx,level in enumerate(self.use_levels):
            if idx == 0:
                img_flat = np.expand_dims(img_dic[level], 0)
            else:
                img_flat = np.concatenate((img_flat, np.expand_dims(img_dic[level], 0)), axis=0)
        for idx,level in enumerate(self.cfg["model"]["use_level_to_cal"][self._way]):
            if idx == 0:
                label_flat = label_dic[level]
            else:
                label_flat = np.append(label_flat,label_dic[level])

        if "GoogLeNet" in self.cfg["model"]["model_name"]:
            for idx, grid in enumerate(img_flat):
                # print(np.resize(grid,(3,299,299)).shape, "0")
                img_flat_[idx] = np.resize(grid, (3,299,299))
            return (img_flat_, label_flat, 0, img_name)
        else:
            return (img_flat, label_flat, 0, img_name)



## debug
if __name__=="__main__":
    dataset_test = ImageFolderDataset(
        # root="/home/server/Disk3/SegmentData/LC25000/lung_colon_image_set/colon_image_sets/val",
        root="/home/server/Disk3/SegmentData/DDSNet/test",
        img_size=512,
        patch_size=256,
        crop_size=224,
        way="train",
        stain_normalizer=False,
        cfg={
            "model": {
                "model_name": "resnet34",
                "use_level_to_cal": {
                    "train": [0],
                    "test": [0],
                    "val": [0],
                }
            }
        },
        total_labels={"no_cancer": 0, "cancer": 1}
    )
    from torch.utils.data import DataLoader

    # print("len(dataset_train): ",len(dataset_train))
    print("len(dataset_test): ",len(dataset_test))
    dataloader_train = DataLoader(dataset_test,
                                  batch_size=64,
                                  num_workers=0,
                                  drop_last=False,
                                  shuffle=True)
    for i in dataset_test:
        print(i[3])
