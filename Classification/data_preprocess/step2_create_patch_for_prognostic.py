from skimage.transform.integral import integral_image, integrate
import cv2
import numpy as np
import os
import json
import argparse
from openslide import OpenSlide
from tqdm import tqdm

from Classification.data_preprocess.utils.make_tissue_mask import choose_level, make_mask


def parse_args():
    parser = argparse.ArgumentParser(description="Go process data")
    #config
    parser.add_argument('--data_root_dir', type=str, default="/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/prognostic")
    parser.add_argument('--prognostic_root_dir', type=str, default="/media/whisper/Newsmy/WSI_PatchData/CY_SL_FD_HS/RFS_Data_UpLoad")
    parser.add_argument('--num_process', type=int, default=30)
    parser.add_argument('--check', type=bool, default=True)

    return parser.parse_args()

def load_mapping_path(data_root_dir, iset):
    with open(f'{data_root_dir}/rfs_{iset}.json', 'r',
              encoding='utf-8') as load_f:
        load_values = json.load(load_f)
    return load_values
if __name__ == '__main__':
    args = parse_args()
    data_root_dir = args.data_root_dir
    prognostic_root_dir = args.prognostic_root_dir
    # dataset = args.dataset

    num_process = args.num_process
    check = args.check

    mask_downsample = 64
    patch_size = 768
    stride = patch_size

    # datasets = ["YouAn", "hz", "huashan2", "huashan", "CY"]
    datasets = ["YouAn"]

    for dataset in datasets:
        ##################################### 0. load json data
        HE_list = load_mapping_path(prognostic_root_dir, dataset)
        print(f"length of {dataset}:{len(HE_list)}")

        file_path_json = os.path.join(data_root_dir, f"{dataset}_patch_cord.json")

        undo_txt = open(f"./{dataset}_wrong_WSI.txt", 'w+')

        json_data = {}

        for iHE_info in tqdm(HE_list):
            if dataset == "YouAn":
                HE_file = os.path.join(data_root_dir, iHE_info['path'], f"{iHE_info['path'].split('+')[1]}.mrxs")
                HE_filepath = os.path.join(iHE_info['path'], f"{iHE_info['path'].split('+')[1]}.mrxs")
                if not os.path.exists(HE_file):
                    undo_txt.writelines([HE_file, '\n'])
                    continue
                # HE_file = os.path.join(data_root_dir, f"YouAn/2022.12.5肝移植2+1M23_06.12.2022_11.33.35/1M23_06.12.2022_11.33.35.mrxs")
                # HE_filepath = os.path.join("2022.12.5肝移植2+1M23_06.12.2022_11.33.35", "1M23_06.12.2022_11.33.35.mrxs")
                slide = OpenSlide(HE_file)
            elif dataset == "huashan2":
                try:
                    HE_file = os.path.join(data_root_dir, f"{iHE_info['path']}.tif")
                    HE_filepath = f"{iHE_info['path']}.tif"
                    slide = OpenSlide(HE_file)
                except:
                    HE_file = os.path.join(data_root_dir, f"{iHE_info['path']}.svs")
                    HE_filepath = f"{iHE_info['path']}.svs"
                    slide = OpenSlide(HE_file)
                finally:
                    if not os.path.exists(HE_file):
                        undo_txt.writelines([HE_file, '\n'])
                        continue
            else:
                HE_file = os.path.join(data_root_dir, f"{iHE_info['path']}.tif")
                HE_filepath = f"{iHE_info['path']}.tif"
                if not os.path.exists(HE_file):
                    undo_txt.writelines([HE_file, '\n'])
                    continue
                slide = OpenSlide(HE_file)

            # ##################################### 2.slide
            down_sample_img = choose_level(slide)
            mask_down_sample_img = make_mask(down_sample_img, save_location=None)
            ##################################### 4. creat patch mapping txt file
            integral_image_tissue = integral_image(mask_down_sample_img.T / 255)
            slide_w_lv_0, slide_h_lv_0 = slide.dimensions
            slide_w_downsample = slide_w_lv_0 / mask_downsample
            slide_h_downsample = slide_h_lv_0 / mask_downsample
            # 768/64
            size_patch_lv_k = int(patch_size / mask_downsample)  # patch在第mask_level层上映射的大小

            # 建立一个树形结构的轮廓，仅保存4点信息
            contours_lab, _ = cv2.findContours(mask_down_sample_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            p_left = []
            p_right = []
            p_bottom = []
            p_top = []
            # 提取tissue区域的随机patch，contours_lab为一个轮廓列表
            for contour in contours_lab:
                # 改变数组的维数，把维度为1的去掉
                coordinates = (np.squeeze(contour)).T
                coords_x = coordinates[0]
                coords_y = coordinates[1]
                # patch的四个顶点
                p_left.append(np.min(coords_x))
                p_right.append(np.max(coords_x))
                p_top.append(np.min(coords_y))
                p_bottom.append(np.max(coords_y))

            stride_lv = int(stride / mask_downsample)
            # print(stride_lv)
            list_type = []
            for contour_idx in range(len(contours_lab)):
                p_x_left = p_left[contour_idx]
                p_x_right = p_right[contour_idx]
                p_y_top = p_top[contour_idx]
                p_y_bottom = p_bottom[contour_idx]
                for x in range(p_x_left, p_x_right, stride_lv):
                    for y in range(p_y_top, p_y_bottom, stride_lv):
                        if (y + size_patch_lv_k > slide_h_downsample) or \
                                (x + size_patch_lv_k > slide_w_downsample):
                            continue
                        # 求解积分
                        tissue_integral = integrate(
                            integral_image_tissue, (x, y), (x + size_patch_lv_k - 1, y + size_patch_lv_k - 1)
                        )
                        tissue_ratio = tissue_integral / (size_patch_lv_k ** 2)
                        if tissue_ratio < 0.7:
                            continue
                        list_type.append([x, y])

            center_x_y_list = []
            for i, item in enumerate(list_type):
                x = item[0]
                y = item[1]
                # center_coord
                patch_x_lv_0 = round(int(x + size_patch_lv_k / 2) * mask_downsample)
                patch_y_lv_0 = round(int(y + size_patch_lv_k / 2) * mask_downsample)
                # print(iHE_info['PID']+" "+ patch_x_lv_0+" "+patch_y_lv_0 + " " + "tissue")
                # imgname = iHE_info['PID'] + "_" + patch_x_lv_0 + '_' + patch_y_lv_0
                center_x_y_list.append([patch_x_lv_0, patch_y_lv_0])
                # txt.writelines([imgname, ',', iHE_info['PID'], ',', patch_x_lv_0, ',', patch_y_lv_0, ',', HE_filepath, '\n'])

            iHE_info['center_x_y_list'] = center_x_y_list
            json_data[HE_filepath] = iHE_info

            if check:
                overlay = down_sample_img.copy()

                cv2.imwrite("./mask.png", mask_down_sample_img)
                for i, item in enumerate(list_type):
                    x = item[0]
                    y = item[1]
                    cv2.rectangle(down_sample_img, (x, y), (x + size_patch_lv_k, y + size_patch_lv_k), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.3, down_sample_img, 1 - 0.3, 0, down_sample_img)
                cv2.imwrite("./check.png", down_sample_img)
        assert len(json_data) == len(HE_list), 'unmatch with rfs data'
        json_str = json.dumps(json_data, ensure_ascii=False, indent=4)
        json_file = open(file_path_json, 'w+', encoding="utf-8")
        with json_file as f:
            f.write(json_str)
