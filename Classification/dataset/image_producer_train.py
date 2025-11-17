# -*-coding:utf-8-*-
import os
import sys
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from Classification.dataset.annotation import Annotation
from PIL import ImageFile
import random
from os.path import getsize
from openslide import OpenSlide
import cv2
import json
from tqdm import tqdm
import logging


ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(0)


class GridImageDataset(Dataset):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    """

    def __init__(self, img_size, patch_size,
                 crop_size=224, normalize=True, way="train", key_word="",
                 sample_class_num=(), is_test=False, tif_dict={},
                 stain_normalizer=False, cfg=None, level=0,
                 use_levels=[0]):
        """
        Initialize the data producer.

        Arguments:
            data_path: string, path to pre-sampled images
            img_size: int, size of pre-sampled images, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
        """
        self._img_size = img_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._key_word = key_word
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        self._way = way
        self.sample_class_num = sample_class_num
        self.is_test = is_test
        self.stain_normalizer = stain_normalizer
        self.cfg = cfg
        self._level = level
        self.total_tifs = tif_dict
        self._annotations = {}
        self.use_levels = use_levels
        self._preprocess()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop((768, 768)),
        ])

    def _preprocess(self):
        # {name:, wsi_path:, x_center:, y_center:, class:, grid_size:, level: }
        
        if self._img_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._img_size, self._patch_size))

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

        # 在create_patch中，按最大最小坐标选的图像块，图像块中可能包括非目标区域，所以这里要读json文件重新确定一下各区域的label标签
        for HE_name, HE_info in self.total_tifs.items():
            anno = Annotation()
            wsi_json_path = os.path.join(self.cfg["dataset"]["data_root"], HE_info["seg_filepath"])
            wsi_path = os.path.join(self.cfg["dataset"]["data_root"], HE_info["HE_filepath"])
            anno.from_json(wsi_json_path)
            self._annotations[HE_name] = anno
            self.total_tifs[HE_name] = OpenSlide(wsi_path)
            print(f"{self._way} HE:{HE_info['HE_filepath']}")
        self._coords = []
        for itype in self.sample_class_num:
            type_patch_txt_path = os.path.join(
                self.cfg["dataset"]["data_root"],
                self.cfg["dataset"][self._way]["patch_cls_path"],
                f"{itype}.txt"
            )
            f_list = open(type_patch_txt_path)
            lines = f_list.readlines()
            for line_idx, line in enumerate(lines):
                img_name, pid, x_center, y_center, img_label = line.strip('\n').split(',')
                ### fuck###
                # if "D17-02013-01" in img_name and self._way=="train":
                #     continue
                # if line_idx % 2 != 0 and img_label == 'cancer' and self._way=="train":
                #     print("FUC222K")
                #     continue
                if pid in self._annotations.keys():
                # if os.path.getsize(
                #     os.path.join(self.cfg["dataset"]["data_root"], data_dict[pid]['HE_filepath'])
                # ) / 1024.0 < 800:
                #     continue
                    self._coords.append(
                        (pid, x_center, y_center, img_label)
                    )
        
    def __len__(self):
        return len(self._coords)

    def __getitem__(self, idx):
        pid, x_center, y_center, kind = self._coords[idx]
        img_name = pid + '_' + str(x_center) + '_' + str(y_center)

        x_top_left = int(float(x_center) - self._img_size / 2)
        y_top_left = int(float(y_center) - self._img_size / 2)

        # the grid of labels for each patch
        label_grid_temp = np.zeros((self._patch_per_side, self._patch_per_side),
                              dtype=np.float32)
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                x = x_top_left + int((x_idx + 0.5) * self._patch_size)
                y = y_top_left + int((y_idx + 0.5) * self._patch_size)
                polygon_dic = {}
                flag = False
                if self._way in ["val", "test"]:
                    if kind == "cancer":
                        label_grid_temp[y_idx, x_idx] = 1
                    else:
                        label_grid_temp[y_idx, x_idx] = 0
                else:
                    for k in self.cfg["dataset"]["sample_class_num"]:
                        coord_info = self._annotations[pid].inside_polygons(k, (x, y))
                        if coord_info[0]:
                            len_polygon = coord_info[1].length()
                            polygon_dic[k] = len_polygon
                            flag = True
                    if flag:
                        polygon_dic = {k:v for k,v in sorted(polygon_dic.items(),key=lambda x:x[1])}
                        if list(polygon_dic)[0] == "cancer":
                            label_grid_temp[y_idx, x_idx] = 1
                        else:
                            label_grid_temp[y_idx, x_idx] = 0
                    else:
                        label_grid_temp[y_idx, x_idx] = 0
                    
                # if kind == 'cancer':
                #     label_grid[y_idx, x_idx] = 1
                # else:
                #     label_grid[y_idx, x_idx] = 0
        img_dic = {}
        label_dic = {}
        for level in self.use_levels:
            # x_range, y_range = self.total_tifs[pid].dimensions
            shift_x_top_left = max(0, x_top_left - (self._patch_size*2**level - self._patch_size)//2)
            shift_y_top_left = max(0, y_top_left - (self._patch_size*2**level - self._patch_size)//2)
            img_dic[level] = self.total_tifs[pid].read_region(
                (shift_x_top_left, shift_y_top_left), level, (self._img_size, self._img_size)).convert('RGB')
        
        for level, img in img_dic.items():
            label_grid = np.copy(label_grid_temp)
            if self.stain_normalizer != False:
                img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Standardize brightness (optional, can improve the tissue mask calculation)
                img = staintools.LuminosityStandardizer.standardize(img)
                # Stain normalize
                img = self.stain_normalizer.transform(img)
                img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


            # color jitter
            if self._way == "train":
                img = self._color_jitter(img)

                # use left_right flip
                if not self.cfg["model"]["concat_level_feats"]:
                    if np.random.rand() > 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        label_grid = np.fliplr(label_grid)

                    # use rotate，进行随机旋转，旋转90*num_rotate度
                    num_rotate = np.random.randint(0, 4)
                    img = img.rotate(90 * num_rotate)
                    label_grid = np.rot90(label_grid, num_rotate)

            # PIL image:   H x W x C
            # torch image: C X H X W
            img = self.transform(img) #将1024*1024的图片随机裁剪成了768*768
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
            return (img_flat_, label_flat, pid, img_name)
        else:
            return (img_flat, label_flat, pid, img_name)

## debug
if __name__=="__main__":
        from torch.utils.data import DataLoader
        import time
        # target = staintools.read_image("/home/data/huangxiaoshuang/test/2/target/target.jpg")
        # target = staintools.LuminosityStandardizer.standardize(target)
        # stain_nomalizer = staintools.StainNormalizer(method='vahadane')
        # stain_nomalizer.fit(target)
        cfg = {
            "dataset": {
                "data_root": "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/",
                "train": {
                    "sample_class_num":{
                        "cancer": 1500,
                        "cancer_beside": 200,
                        "normal_liver": 200,
                        "hemorrhage_necrosis": 200,
                        "tertiary_lymphatic": 200,
                        "other": 31
                    },
                    "data_path": "Annotation/segmentation/HE2Json_train.json",
                    "patch_cls_path": "segmentation/patch_cls/train/"
                }
            },
            "model": {
                "concat_level_feats": False,
                "model_name": "resnet34",
            }
        }
        dataset_train = GridImageDataset("/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Annotation/segmentation/HE2Json_train.json",
                                        768,
                                        256,
                                        224,
                                        sample_class_num=cfg['dataset']["train"]["sample_class_num"],
                                         cfg=cfg
                                         )

        dataloader_train = DataLoader(dataset_train,
                                        batch_size=28,
                                        num_workers=0,
                                        drop_last=False,
                                        shuffle=True)
        for step,tmp in enumerate(dataloader_train):
            time_now = time.time()
            print(time_now)
            data, target, _, _ = tmp
            print(time_now)