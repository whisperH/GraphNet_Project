import cv2
import numpy as np

import os
import sys
from os import listdir

from openslide import OpenSlide
import json as js
import struct
import collections
import time
from multiprocessing import Pool

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
np.set_printoptions(threshold=sys.maxsize)


def find_contours_of_xml(cur_path_json, downsample):
    # 先把json文件中的颜色信息取出 #f5a623,contours是字典类型
    contours = {}
    Lable = []
    # 这边加一句,encoding='utf-8'
    with open(cur_path_json, 'r', encoding='utf-8') as load_f:
        # 将数据读入
        data = js.load(load_f)

    Models =  data['Models']
    PolygonModel2 = Models['PolygonModel2']
    contours = collections.defaultdict(list)
    for Lesion in PolygonModel2:
        if Lesion['Label'] not in Lable:
            Lable.append(Lesion['Label'])
    for cval in Lable:
        for Lesion in PolygonModel2:
            list_blob = []
            if Lesion['Label'] == cval:

                for value in Lesion['Points']:
                    list_point = []
                    try:
                        p_x = value[0]
                        p_y = value[1]
                        p_x = p_x / downsample
                        p_y = p_y / downsample
                        list_point.append([int(round(p_x)), int(round(p_y))])

                        if len(list_point) >= 0:
                            list_point = np.array(list_point, dtype=np.int32)
                            list_blob.append(list_point)

                    except:
                        continue
                list_blob_new = []
                for list_point in list_blob:
                    list_point_int = [[[int(round(point[0])), int(round(point[1]))]] \
                                      for point in list_point]
                    list_blob_new.append(list_point_int)
                contour = np.array(list_blob_new, dtype=np.int32)
                con = []
                for c in contour:
                    con.append(c[0])
                contour_new = np.array(con, dtype=np.int32)

                contours[cval].append(contour_new)



    return contours


# def hex2rgb(hex_str):
#     print(hex_str)
#     if hex_str == 3:
#         hex_str = "#0000ff"
#         # label2是癌组织（绿色）
#     if hex_str == 2:
#         hex_str = "#00ff00"
#     if hex_str == 5:
#         hex_str = "#00ffff"
#     str = hex_str.split('#')[-1]
#     int_tuple = struct.unpack('BBB', bytes.fromhex(str))
#     return tuple([val for val in int_tuple])

def hex2rgb(hex_str):
    str = hex_str.split('#')[-1]
    int_tuple = struct.unpack('BBB', bytes.fromhex(str))
    return tuple([val for val in int_tuple])


def make_tumor_mask(mask_shape, contours):
    wsi_empty = np.zeros(mask_shape[:2])

    wsi_empty = wsi_empty.astype(np.uint8)
    # 这去掉就没东西了
    cv2.drawContours(wsi_empty, contours, -1, (255, 255, 255), -1)
    return wsi_empty


# 获取标签
def get_label_list(label, dict):
    label_list = []
    for key, value in dict.items():
        if key == label:
            label_list.append(key)
    for key, value in dict.items():
        if key != label:
            label_list.append(key)
    return label_list


# origin就是之前划分出来的img
def save_res_mask(
        dict,
        cur_path_origin,
        label_list,
        label,
        save_location_path):
    wsi_bgr_jpg = cv2.imread(cur_path_origin)
    wsi_jpg_vi = wsi_bgr_jpg.copy()

    for val in label_list:
        # rgb_color = hex2rgb(val)
        # bgr_color = (rgb_color[-1], rgb_color[1], rgb_color[0])
        if val == label:
            cv2.drawContours(wsi_jpg_vi,
                             dict.get(val),
                             -1,
                             # 这边的颜色是轮廓内的填充颜色，改成了(0,0,0)
                             #                            bgr_color,
                             (0, 0, 0),
                             -1)
        else:
            cv2.drawContours(wsi_jpg_vi,
                             dict.get(val),
                             -1,
                             (255, 255, 255),
                             -1)

    wsi_gray_lv_ = cv2.cvtColor(wsi_jpg_vi, cv2.COLOR_BGR2GRAY)
   
    ret, wsi_bin_0255_lv_ = cv2.threshold(
        wsi_gray_lv_,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_o = np.ones((2, 2), dtype=np.uint8)
    kernel_c = np.ones((4, 4), dtype=np.uint8)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
        wsi_bin_0255_lv_, \
        cv2.MORPH_CLOSE, \
        kernel_c)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
        wsi_bin_0255_lv_, \
        cv2.MORPH_OPEN, \
        kernel_o)

    contours_tissue_lv_, hierarchy = \
        cv2.findContours( \
            wsi_bin_0255_lv_, \
            cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_SIMPLE)

    mask_shape_lv_ = wsi_gray_lv_.shape
    
    tissue_mask_lv_ = make_tumor_mask(mask_shape_lv_, contours_tissue_lv_)

    tumor_mask_0255 = make_tumor_mask(mask_shape_lv_, dict.get(label))

    and_mask = cv2.bitwise_and(tissue_mask_lv_, tumor_mask_0255)

    contours2, h2 = cv2.findContours(
        and_mask,
        cv2.RETR_CCOMP,
        2)

    hierarchy = np.squeeze(h2)
    if len(hierarchy.shape) == 1:
        if (hierarchy[3] != -1):
            cv2.drawContours(tumor_mask_0255, contours2, -1, (0, 0, 0), -1)
    else:

        for i in range(len(contours2)):

            if (hierarchy[i][3] != -1):
                cv2.drawContours(tumor_mask_0255, contours2, i, (0, 0, 0), -1)
    print("save_location_path: "+str(save_location_path))

    cv2.imwrite(save_location_path, tumor_mask_0255)


def make_mask(
        dict,
        cur_path_origin,
        save_other,
        save_cancer_mask,
        save_normal_liver_mask,
        save_cancer_beside_mask,
        save_hemorrhage_necrosis,
        save_tertiary_lymphatic,
        save_filename
):
    for label in dict.keys():  # label 3 蓝色圈内的是正常肝组织   label 2 绿色的是癌组织 label 5 间质
        label_list = get_label_list(label, dict)
        save_location = ''

        # label1是其他
        if label == 1:
            save_location = save_other
        # label2是癌组织（绿色）
        elif label == 2:
            save_location = save_cancer_mask
        # label3是正常肝组织（深蓝色）
        elif label == 3:
            save_location = save_normal_liver_mask
        # lable 4 出血坏死
        elif label == 4:
            save_location = save_hemorrhage_necrosis
        # label 5 间质
        elif label == 5:
            save_location = save_cancer_beside_mask
        # lable 6 三级淋巴结构
        elif label == 6:
            save_location = save_tertiary_lymphatic
        else:
            continue


        save_mask_and_path = os.path.join(save_location, save_filename)
        save_res_mask(
            dict,
            cur_path_origin,
            label_list,
            label,
            save_mask_and_path)



def make_lab_mask(data_root_dir, tiff_path, save_path_jpg, save_lab_mask):
    save_other = os.path.join(save_lab_mask, 'other/')
    save_cancer_mask = os.path.join(save_lab_mask, 'cancer/')
    save_normal_liver_mask = os.path.join(save_lab_mask, 'normal_liver/')
    save_cancer_beside_mask = os.path.join(save_lab_mask, 'cancer_beside/')
    save_hemorrhage_necrosis = os.path.join(save_lab_mask, 'hemorrhage_necrosis/')
    save_tertiary_lymphatic = os.path.join(save_lab_mask, 'tertiary_lymphatic')
    os.makedirs(save_other, exist_ok=True)
    os.makedirs(save_cancer_mask, exist_ok=True)
    os.makedirs(save_normal_liver_mask, exist_ok=True)
    os.makedirs(save_cancer_beside_mask, exist_ok=True)
    os.makedirs(save_hemorrhage_necrosis, exist_ok=True)
    os.makedirs(save_tertiary_lymphatic, exist_ok=True)
    print('start')
    t1 = time.time()
    downsample = 64
    opt_list = []
    for file_name, file_info in tiff_path.items():
        cur_file_path = os.path.join(data_root_dir, file_info['HE_filepath'])
        cur_path_json = os.path.join(data_root_dir, file_info['seg_filepath'])
        dict = find_contours_of_xml(cur_path_json, downsample)
        cur_path_origin = os.path.join(save_path_jpg, file_name + '_origin_cut_64.png')
        save_filename = file_name + '_mask_64.png'
        print("save_filename是：" + str(save_filename))
        # dict即整数坐标列表
        opt_list.append((
            dict,
            cur_path_origin,
            save_other,
            save_cancer_mask,
            save_normal_liver_mask,
            save_cancer_beside_mask,
            save_hemorrhage_necrosis,
            save_tertiary_lymphatic,
            save_filename))

    pool = Pool(5)
    pool.starmap(make_mask, opt_list)
    pool.close()
    pool.join()

    t2 = time.time()
    print((t2 - t1) / 60)
    print('end')

if __name__ == '__main__':

    # tiff_path = '/home/data/medicine/20221012_3tif/tif'
    # file_path_json = '/home/data/medicine/20221012_3tif/json'
    # file_path_origin = '/home/data/medicine/20221012_3tif_768_all_cls_tissue06_lab08_v1/img'
    # save_lab_mask ="/home/data/medicine/20221012_3tif_768_all_cls_tissue06_lab08_v1/lab_mask/"
    #

    data_root_dir = "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS"
    tiff_path = {
        "9123526T 杨邦宏 T HE": {
            "HE_filepath": "segmentation/WSI_Data/9123526T 杨邦宏 T HE.tif",
            "file_suffix": ".tif",
            "seg_filepath": "Annotation/segmentation/GT/9123526T 杨邦宏 T HE_tif_Label.json"
        },
        "9187359 康涛 癌 HE": {
            "HE_filepath": "segmentation/WSI_Data/9187359 康涛 癌 HE.tif",
            "file_suffix": ".tif",
            "seg_filepath": "Annotation/segmentation/GT/9187359 康涛 癌 HE_tif_Label.json"
        }
    }
    save_path_jpg = f"{data_root_dir}/segmentation/img/"
    save_lab_mask = f"{data_root_dir}/segmentation/lab_mask/"
    # make_lab_mask(data_root_dir, tiff_path, save_path_jpg, save_lab_mask)

    for file_name, file_info in tiff_path.items():
        cur_file_path = os.path.join(data_root_dir, file_info['HE_filepath'])
        cur_path_json = os.path.join(data_root_dir, file_info['seg_filepath'])
        dict = find_contours_of_xml(cur_path_json, downsample=64)
        cur_path_origin = os.path.join(save_path_jpg, file_name + '_origin_cut_64.png')
        save_filename = file_name + '_mask_64.png'
        print("save_filename是：" + str(save_filename))
        make_mask(
            dict,
            cur_path_origin,
            save_other="",
            save_cancer_mask="",
            save_normal_liver_mask="",
            save_cancer_beside_mask="",
            save_hemorrhage_necrosis="",
            save_tertiary_lymphatic="",
            save_filename=""
        )