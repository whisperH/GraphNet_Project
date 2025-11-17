import cv2
import numpy as np
from openslide import OpenSlide
import sys

from multiprocessing import Pool
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')


def save_slide_cutting_multiple(
        file_path,
        save_location,
        multiple

):
    if isinstance(file_path, str):
        print("save_slide_cutting_multiple中的路径  " + file_path)
        # 检验文件是否存在
        print((os.path.exists(file_path)))
        slide = OpenSlide(file_path)
    else:
        slide = file_path
    level_cnt = slide.level_count
    for level in reversed(range(level_cnt)):
        downsample = slide.level_downsamples[level]
        w_lv_, h_lv_ = slide.level_dimensions[level]
        wsi_pil_lv_ = slide.read_region(
            (0, 0),
            level,
            (w_lv_, h_lv_))
        wsi_ary_lv_ = np.array(wsi_pil_lv_)
        wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)
        downsample = round(downsample)
        print(downsample)
        if downsample > multiple:
            continue
        elif downsample < multiple:
            downsample = multiple / downsample
            w = int(w_lv_ / downsample)
            h = int(h_lv_ / downsample)
            img = cv2.resize(wsi_bgr_lv_, (w, h), interpolation=cv2.INTER_LINEAR)
            if save_location is None:
                return img
            else:
                cv2.imwrite(save_location, img)
            break
        else:
            img = wsi_bgr_lv_
            if save_location is None:
                return img
            else:
                cv2.imwrite(save_location, img)
            break


def save_slide_cutting(file_path, save_location, level):
    slide = OpenSlide(file_path)
    print('==> saving slide_lv_' + str(level) + ' at ' + save_location)
    x_lv_, y_lv_ = 0, 0
    w_lv_, h_lv_ = slide.level_dimensions[level]
    try:
        wsi_pil_lv_ = slide.read_region((0, 0), level,
                                        (w_lv_, h_lv_))

        wsi_ary_lv_ = np.array(wsi_pil_lv_)
        wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(save_location, wsi_bgr_lv_)
    except:
        print(file_path)


def make_slide_cutting(data_root_dir, file_path_tif, save_path_jpg, key_word='HE_filepath'):
    os.makedirs(save_path_jpg, exist_ok=True)
    print("start")
    # print("fileName = " + os.path.dirname(os.path.abspath(__file__)) + '/../../')

    multiple = 64
    print("multiple = " + str(multiple))  # 我加的

    for file_name, file_info in file_path_tif.items():
        cur_save_loca = os.path.join(save_path_jpg, f"{file_name}_origin_cut_64.png")
        print(f'save at {cur_save_loca}')
        cur_file_path = os.path.join(data_root_dir, file_info[key_word])
        save_slide_cutting_multiple(cur_file_path, cur_save_loca, multiple)

    print("end")

if __name__ == '__main__':
    # file_path_tif = "/home/data/mass_spectra_image/"
    # save_path_jpg = "/home/guozihao/gzh_medical_image_classify_75tif/img/"
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
    make_slide_cutting(data_root_dir, tiff_path, save_path_jpg)