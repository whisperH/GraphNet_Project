# -*-coding:utf-8-*-
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from Classification.dataset.annotation import Annotation
from PIL import ImageFile
import random
from os.path import getsize
from openslide import OpenSlide

ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(0)


class GridImageDataset(Dataset):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    """

    def __init__(self, data_path, json_path, img_size, patch_size,
                 crop_size=224, normalize=True, way="train",key_word="",
                 sample_class_num={}, valid_number=800,is_test=False):
        """
        Initialize the data producer.

        Arguments:
            data_path: string, path to pre-sampled images
            json_path: string, path to the annotations in json format
            img_size: int, size of pre-sampled images, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
        """
        self._data_path = data_path
        self._json_path = json_path
        self._img_size = img_size
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._key_word = key_word
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        self._way = way
        self.sample_class_num = sample_class_num
        self.is_test = is_test
        self.valid_number = valid_number

        self._preprocess()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop((768, 768)),
        ]
        )

    def _preprocess(self):
        if self._img_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._img_size, self._patch_size))

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side
        # for i in range(len(self._data_path)):

        json_file = os.listdir(self._json_path)
        self._annotations = {}

        for jf in json_file:
            if jf.endswith(".json"):
                jf_path = os.path.join(self._json_path, jf)

                if "classification" in jf:
                        continue
                if "TCGA-2Y-A9H1" in jf:
                    continue
                if 'TCGA-2Y-A9H0' in jf:
                    continue
                if 'TCGA-2Y-A9GW' in jf:
                    continue
                if 'TCGA-BC-A10R' in jf:
                    continue

                pid = jf.split('.json')[0]
                if pid[-10:] == '_tif_Label':
                    pid = pid[:-10]
                pid_json_path = jf_path
                anno = Annotation()

                anno.from_json(pid_json_path)

                self._annotations[pid] = anno


        self._coords = []
        self._pth = []
        count = 0
        if self._way == "train":

                patch_path = self._data_path+ '/' + self._way
                file_patch = os.listdir(patch_path)
                for f in file_patch:

                    f_pth = os.path.join(patch_path, f)
                    if not os.path.isdir(f_pth):
                        continue
                    #old or new
                    if f == 'train_cancer' or f == 'cancer':
                        sample = self.sample_class_num['cancer']
                    elif f == 'train_cancer_beside' or f == 'cancer_beside':
                        sample = self.sample_class_num['cancer_beside']
                    elif f == 'train_hemorrhage_necrosis' or f == 'hemorrhage_necrosis':
                        sample = self.sample_class_num['hn']
                    elif f == 'train_tertiary_lymphatic' or f == 'tertiary_lymphatic':
                        sample = self.sample_class_num['tl']
                    elif f == 'other':
                        sample = self.sample_class_num['other']
                    else:
                        sample = self.sample_class_num['normal']
                    f_list = open(os.path.join(f_pth, 'list.txt'))
                    lines = f_list.readlines()
                    if len(lines) > sample:
                        line_index = random.sample(range(len(lines)), sample)
                    else:
                        line_index = range(len(lines))
                    
                    print("training class number per epoch: ", f, len(line_index))
                    if len(lines) == 0 or len(lines[0].strip('\n').split(',')) == 4: #old
                        for idx in line_index:
                            pid, x_center, y_center, kind = lines[idx].strip('\n').split(',')[0:4]
                            if 'TCGA-2Y-A9H1' in pid:
                                continue
                            mod = idx // 5000
                            img_name = pid + '_' + str(idx) + '.png'
                            img_pth = f_pth + '/' + str(mod) + '/' + img_name
                            size = getsize(img_pth)
                            size = size / 1024.0
                            if 'necrosis' in img_pth:
                                if size < 500:
                                    continue
                            else:
                                if size < 800:
                                    continue

                            x_center, y_center = int(float(x_center)), int(float(y_center))
                            self._coords.append((pid, x_center, y_center, kind, img_name))
                            self._pth.append(img_pth)
                    else: #new
                        for idx in line_index:
                            img_name, pid, x_center, y_center, kind = lines[idx].strip('\n').split(',')[0:5]
                            # img_name = pid + '_' + str(idx) + '.png'
                            img_pth = os.path.join(f_pth, pid, img_name)
                            size = getsize(img_pth)
                            size = size / 1024.0
                            if 'necrosis' in img_pth:
                                if size < 500:
                                    continue
                            else:
                                if size < 800:
                                    continue

                            x_center, y_center = int(float(x_center)), int(float(y_center))
                            self._coords.append((pid, x_center, y_center, kind, img_name))
                            self._pth.append(img_pth)
                    f_list.close()
                count += 1

        if self._way == "valid":

                valid_pth = self._data_path + '/' + self._way
                for f in os.listdir(valid_pth):
                    if f == 'tertiary_lymphatic':
                        sample = 0
                    elif f == 'other':
                        sample = 0
                    elif self.is_test: # use all test dataset to test
                        sample = 1e+10
                    elif f == 'cancer':# use a little test dataset as the valid dataset to adjust hyper-params
                        sample = self.valid_number
                    elif f == 'cancer_beside':
                        sample = self.valid_number
                    elif f == 'hemorrhage_necrosis':
                        sample = self.valid_number
                    elif f == 'normal_liver':
                        sample = self.valid_number

                    f_pth = os.path.join(valid_pth, f)
                    if not os.path.isdir(f_pth):
                        continue
                    try:
                        file_ = open(os.path.join(f_pth, 'list_uniq.txt'))
                    except:
                        file_ = open(os.path.join(f_pth, 'list.txt'))

                    lines = file_.readlines()
                    if len(lines) > sample:
                        line_index = random.sample(range(len(lines)), sample)
                    else:
                        line_index = range(len(lines))
                    print("valid class number per epoch: ", f, len(line_index))
                    if len(lines) == 0 or len(lines[0].strip('\n').split(',')) == 4:
                        for idx in line_index:

                            pid, x_center, y_center, kind = lines[idx].strip('\n').split(',')[0:4]
                            if 'TCGA-2Y-A9H1' in pid:
                                continue
                            if 'TCGA-2Y-A9H0' in pid:
                                continue
                            if 'TCGA-2Y-A9GW' in pid:
                                continue
                            if 'TCGA-BC-A10R' in pid:
                                continue
                            x_center, y_center = int(float(x_center)), int(float(y_center))

                            mod = idx // 5000
                            img_name = pid + '_' + str(idx) + '.png'
                            img_pth = f_pth + '/' + str(mod) + '/' + img_name
                            size = getsize(img_pth)
                            size = size / 1024.0
                            if 'necrosis' in img_pth:
                                if size < 500:
                                    continue
                            else:
                                if size < 800:
                                    continue
                            self._coords.append((pid, x_center, y_center, kind, img_name))
                            self._pth.append(img_pth)
                    else:
                        for idx in line_index:
                            img_name, pid, x_center, y_center, kind = lines[idx].strip('\n').split(',')[0:5]
                            # img_name = pid + '_' + str(idx) + '.png'
                            img_pth = os.path.join(f_pth, pid, img_name)
                            size = getsize(img_pth)
                            size = size / 1024.0
                            if 'necrosis' in img_pth:
                                if size < 500:
                                    continue
                            else:
                                if size < 800:
                                    continue

                            x_center, y_center = int(float(x_center)), int(float(y_center))
                            self._coords.append((pid, x_center, y_center, kind, img_name))
                            self._pth.append(img_pth)

                    file_.close()


    def __len__(self):
        return len(self._coords)

    def __getitem__(self, idx):
        pid, x_center, y_center, kind, img_name = self._coords[idx]

        x_top_left = int(x_center - self._img_size / 2)
        y_top_left = int(y_center - self._img_size / 2)

        # the grid of labels for each patch
        label_grid = np.zeros((self._patch_per_side, self._patch_per_side),
                              dtype=np.float32)
        num_normal = 0
        i = 0

        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # (x, y) is the center of each patch
                x = x_top_left + int((x_idx + 0.5) * self._patch_size)
                y = y_top_left + int((y_idx + 0.5) * self._patch_size)
                flag = False
                lymph_flag = False
                len_polygon1 = 0
                len_polygon2 = 0
                len_polygon3 = 0
                len_polygon4 = 0

               # if self._annotations[pid].inside_polygons('lymph', (x, y))[0]:
                #     len_polygon1 = self._annotations[pid].inside_polygons('lymph', (x, y))[1].length()
                #     lymph_flag = True
                if self._annotations[pid].inside_polygons('cancer', (x, y))[0]:
                    len_polygon1 = self._annotations[pid].inside_polygons('cancer', (x, y))[1].length()
                    flag = True
                    label = 1
                if self._annotations[pid].inside_polygons('cancer_beside', (x, y))[0]:
                    if flag == True:
                        len_polygon2 = self._annotations[pid].inside_polygons('cancer_beside', (x, y))[1].length()
                        if len_polygon1 > len_polygon2:
                            label = 0
                        else:
                            label = label
                    else:
                        flag = True
                        label = 0
                if self._annotations[pid].inside_polygons('normal_liver', (x, y))[0]:
                    if flag == True:
                        len_polygon3 = self._annotations[pid].inside_polygons('normal_liver', (x, y))[1].length()
                        if len_polygon1 > len_polygon3 or len_polygon2 > len_polygon3:
                            label = 0
                        else:
                            label = label
                    else:
                        flag = True
                        label = 0
                if self._annotations[pid].inside_polygons('hemorrhage_necrosis', (x, y))[0]:
                    label = 0
                if self._annotations[pid].inside_polygons('tertiary_lymphatic', (x, y))[0]:
                    label = 0
                if self._annotations[pid].inside_polygons('other', (x, y))[0]:
                    label = 0
                # if self._annotations[pid].inside_polygons('necrosis', (x, y))[0]:
                #     if flag == True:
                #         len_polygon4 = self._annotations[pid].inside_polygons('necrosis', (x, y))[1].length()
                #         if len_polygon1 > len_polygon4 or len_polygon2 > len_polygon4 or len_polygon3 > len_polygon4:
                #             label = 0
                #         else:
                #             label = label
                #     else:
                #         flag = True
                #         label = 0
                if not (self._annotations[pid].inside_polygons('cancer', (x, y))[0] \
                        or self._annotations[pid].inside_polygons('cancer_beside', (x, y))[0] \
                        or self._annotations[pid].inside_polygons('normal_liver', (x, y))[0] \
                        or self._annotations[pid].inside_polygons('hemorrhage_necrosis', (x, y))[0] \
                        or self._annotations[pid].inside_polygons('tertiary_lymphatic', (x, y))[0] \
                        or self._annotations[pid].inside_polygons('other', (x, y))[0]):
                    label = 0
                    num_normal += 1
                if kind == "normal_liver":
                    label = 0
                elif kind == "cancer":
                    label = 1
                elif kind == "cancer_beside":
                    label = 2
                elif kind == "hemorrhage_necrosis":
                    label = 3
                label_grid[y_idx, x_idx] = label
        try:
            img = Image.open(self._pth[idx])
        except:
            print(self._pth[idx])
        # color jitter
        if self._way == "train":
            img = self._color_jitter(img)

            # use left_right flip
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
        label_flat = np.zeros(self._grid_size, dtype=np.int)

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

        return (img_flat, label_flat, pid, img_name)