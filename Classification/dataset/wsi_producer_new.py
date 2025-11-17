import json

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import openslide
import PIL
from PIL import Image
np.random.seed(0)
import cv2
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor,as_completed, wait, ALL_COMPLETED
from copy import deepcopy

def read_region_area(parameters):
    x, y, slide, image_size, level, num_x, root_path = parameters
    if y == 0 and x % 5 == 0:
        print(f"starting processing region of {x} to {x + 4} Xcolumn of {num_x}....")
        # print("level_:" + str(level), slide.level_dimensions[level])
    grid_pixels = image_size * image_size

    try:
        img = slide.read_region(
            (x * image_size, y * image_size), level,
            (image_size, image_size)).convert('RGB')

        image_transforms = transforms.Compose([transforms.Grayscale(1)])
        tmp_img = np.array(image_transforms(img))
        # if np.sum(tmp_img) < self._image_size*self._image_size*self.grid_thre*255: #没有超过60%为白色区域
        if len(np.where(tmp_img > 210)[0]) / grid_pixels < 0.8:  # 没有超过80%为白色区域
            # print(f"Add {x* image_size}-{y* image_size} to patch list")
            return x * image_size, y * image_size, img
        else:
            # print("jump")
            return None
    except Exception as e:
        print("loading error region")
        return None

class GridWSIPatchDataset(Dataset):
    """
    Data producer that generate all the square grids, e.g. 3x3, of patches,
    from a WSI and its tissue mask, and their corresponding indices with
    respect to the tissue mask
    """
    def __init__(self, root, wsi_name, mask_path=None, image_size=768, patch_size=256,
                 crop_size=224, normalize=True, flip='NONE', rotate='NONE', level=0,
                 grid_thre=0.6,use_levels=[0]):
        """
        Initialize the data producer.

        Arguments:
            wsi_path: string, path to WSI file
            mask_path: string, path to mask file in numpy format
            image_size: int, size of the image before splitting into grid, e.g.
                768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
            flip: string, 'NONE' or 'FLIP_LEFT_RIGHT' indicating the flip type
            rotate: string, 'NONE' or 'ROTATE_90' or 'ROTATE_180' or
                'ROTATE_270', indicating the rotate type
        """
        self._root = root
        self._wsi_name = wsi_name
        self._mask_path = mask_path
        self._image_size = image_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._flip = flip
        self._level = level
        self._rotate = rotate
        self._patch_list = []
        self.grid_thre = grid_thre
        self.use_levels = use_levels

        self._patch_per_side = self._image_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side


        # self._preprocess()
        self._multi_preprocess()

    def _preprocess(self):
        print("tif_path:", self._root)
        self._slide = openslide.OpenSlide(self._root)
        X_slide, Y_slide = self._slide.level_dimensions[self._level]
        print(self._slide.level_dimensions)
        print("level_:"+str(self._level), self._slide.level_dimensions[self._level])
        num_x = X_slide // self._image_size
        num_y = Y_slide // self._image_size
        cout = 0
        grid_pixels = self._image_size * self._image_size 
        for x in tqdm(range(num_x)):
            for y in range(num_y):
                tmp_img = self._slide.read_region(
                            (x*self._image_size, y*self._image_size), self._level, (self._image_size, self._image_size)).convert('RGB')
                image_transforms = transforms.Compose([transforms.Grayscale(1)])
                tmp_img = np.array(image_transforms(tmp_img))
                # if np.sum(tmp_img) < self._image_size*self._image_size*self.grid_thre*255: #没有超过60%为白色区域
                if len(np.where(tmp_img>210)[0])/grid_pixels < 0.8: #没有超过80%为白色区域
                    self._patch_list.append([x*self._image_size, y*self._image_size])
                    cout += 1
                    # print(cout)
            #         if cout >= 10:
            #             break
            # if cout >= 10:
            #     break
        print('scanning nums:', cout)
        print(self._wsi_name + " patch_nums---->", len(self._patch_list))
        


    def _multi_preprocess(self):
        # print("tif_path:", self._root)
        self._slide = openslide.OpenSlide(self._root)

        X_slide, Y_slide = self._slide.level_dimensions[self._level]

        num_x = X_slide // self._image_size
        num_y = Y_slide // self._image_size
        process_list = []
        region_list = []
        for x in range(num_x):
            for y in range(num_y):
                process_list.append(
                    [x, y, self._slide, self._image_size, self.use_levels[0], num_x, self._root]
                )
            # if len(process_list)>1000:
            #     break

        with ThreadPoolExecutor() as pool:
            all_task = []
            for iprocess in process_list:
                all_task.append(
                    pool.submit(read_region_area, iprocess)
                )
            # wait(all_task, return_when=ALL_COMPLETED)
            for i in as_completed(all_task):
                if i.result() is not None:
                    region_list.append(i.result())

        self._patch_list = region_list

        print(f'scanning nums: {len(process_list)}/{len(self._patch_list)}', )

    def __len__(self):
        return len(self._patch_list)

    def __getitem__(self, idx):
        x, y, img = self._patch_list[idx]
        # print(self._slide.level_dimensions)
        try:
            # print(x, y)
            # img = self._slide.read_region((x, y), self._level, (self._image_size, self._image_size)).convert('RGB')

            #### vis patches ####
            # img.save("/home/huangxiaoshuang/medicine/HCC_Prognostic/Classification/dataset/tmp/"+self._wsi_name+"_"+str(x)+"_"+str(y)+".png", 'PNG')

            # PIL image:   H x W x C
            # torch image: C X H X W
            img = np.array(img, dtype=np.float32).transpose((2, 0, 1))
            if self._normalize:
                img = (img - 128.0)/128.0

            img_dic = {}

            # flatten the square grid
            img_flat = np.zeros(
                (self._grid_size, 3, self._crop_size, self._crop_size),
                dtype=np.float32)

            source_img_flat = np.zeros(
                (self._grid_size, 3, self._patch_size, self._patch_size),
                dtype=np.float32)

            x_set = np.zeros((self._grid_size), dtype=np.int32)
            y_set = np.zeros((self._grid_size), dtype=np.int32)

            idx = 0
            level = 0
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
                    # gap = (self._patch_size - self._crop_size) // 2
                    # source_img_flat[idx] = img[:, (x_start - gap):(x_end + gap), (y_start - gap):(y_end + gap)]
                    # x_set[idx] = x + (x_start - gap)
                    # y_set[idx] = y + (y_start - gap)
                    source_img_flat[idx] = img[:, self._patch_size*x_idx:(self._patch_size*x_idx + self._patch_size), self._patch_size*y_idx:(self._patch_size*y_idx + self._patch_size)]

                    # patch坐标倍率
                    ratio_coord = 2 ** self._level
                    x_set[idx] = x + self._patch_size*y_idx*ratio_coord
                    y_set[idx] = y + self._patch_size*x_idx*ratio_coord
                    idx += 1

            img_dic[level] = img_flat

            for idx,level in enumerate(self.use_levels):
                if idx == 0:
                    img_flat = np.expand_dims(img_dic[level], 0)
                else:
                    img_flat = np.concatenate((img_flat, np.expand_dims(img_dic[level], 0)), axis=0)
            # print(img_flat.shape, source_img_flat.shape)
            # (1, 9, 3, 224, 224) (9, 3, 256, 256)

            return (img_flat, self._wsi_name, x_set, y_set, source_img_flat, self._root, img)
        except Exception as e:
            print(e)
            exit(11111)
            # print(e)
            # return None, None, None, None, None, None, None
if __name__ == "__main__":
    # dataset_valid = GridWSIPatchDataset("/media/server/Newsmy/WSI_RawData/CY_SL_FD_HS/prognostic/huashan/D16-01976-10 1.tif",
    #                                 "D16-01976-10 1",
    #                             level=0,)
    #
    # dataloader_valid = DataLoader(dataset_valid,
    #                             batch_size=1,
    #                             num_workers=0,
    #                             drop_last=False,
    #                             shuffle=False)
    # for idx, data in enumerate(dataloader_valid):
    #     img, name, x, y, source_img_, root, source_img = data
    #     print(img.shape, name, x, y, source_img_.shape, source_img.shape)
    #     source_img = source_img[0].numpy().transpose((1, 2, 0))
    #     source_img = source_img[:, :, ::-1]*128 + 128
    #     cv2.imwrite("./source_img.png", source_img)
    #     for i in range(9):
    #         path = "../work_dirs/"+name[0]+"_"+str(x[0][i].item())+"_"+str(y[0][i].item())+".png"
    #         source_img = source_img_[0][i]
    #         source_img = source_img.numpy().transpose((1, 2, 0))
    #         source_img = source_img[:, :, ::-1]*128 + 128
    #         cv2.imwrite(path, source_img)
    #     break
    from openslide.deepzoom import DeepZoomGenerator

    grid_pixels = 768 * 768
    slide = openslide.OpenSlide("/media/server/Newsmy/WSI_RawData/CY_SL_FD_HS/MoleculeValidation/ZY4HE26605ZHL/D16-00306-03 ①.svs")
    data_gen = DeepZoomGenerator(slide, tile_size=768, overlap=0, limit_bounds=True)
    level = data_gen.level_count - 1
    [cols, rows] = data_gen.level_tiles[-1]
    print(cols, rows)
    for col in range(cols):
        for row in range(rows):
            print("Processing %s %s" % (col, row))
            img = data_gen.get_tile(level, (col, row))
            # 如果图像是纯白，则跳过
            image_transforms = transforms.Compose([transforms.Grayscale(1)])
            tmp_img = np.array(image_transforms(img))
            # if np.sum(tmp_img) < self._image_size*self._image_size*self.grid_thre*255: #没有超过60%为白色区域
            if len(np.where(tmp_img > 210)[0]) / grid_pixels < 0.8:  # 没有超过80%为白色区域
                # print(f"Add {x* image_size}-{y* image_size} to patch list")
                print(col*768, row*768)
                # 保存图像到 output 文件夹（需要提前创建这个文件夹）
                img.save("../work_dirs/D19-01394-04_%s_%s.jpg" % (col*768, row*768))
