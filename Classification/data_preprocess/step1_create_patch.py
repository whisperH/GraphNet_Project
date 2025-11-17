import utils.make_lab_mask as make_lab_mask
import utils.make_slide_cutting as make_slide_cutting
import utils.make_tissue_mask as make_tissue_mask
import utils.create_patch as create_patch
import numpy as np
import os
import json
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Go process data")
    #config
    parser.add_argument('--data_root_dir', type=str, default="/media/server/Newsmy/WSI_RawData/CY_SL_FD_HS")
    parser.add_argument('--num_process', type=int, default=30)
    parser.add_argument('--config_file', type=str, default="./new_data_process/config.json")
    #input dir
    parser.add_argument('--tiff_path', type=str, default="./data2/tif/",
                        help="in path ,the tif files dir")
    parser.add_argument('--file_path_json', type=str, default="./data2/json/",
                        help="in path ,the json files dir") #处理的是id（与tif文件名同名）+.json（扩展名）的文件或者id+ "_tif_Label" + ".json"的文件名。如：D14-01462-01.json或者D14-01462-01_tif_Label.json
    #output dir
    parser.add_argument('--save_tissue_mask', type=str, default="./data2/tissue_mask/",
                        help="output path ,the tissue mask files save dir")
    parser.add_argument('--save_path_jpg', type=str, default="./data2/img/",
                        help="output path ,the jpg save files dir")
    parser.add_argument('--save_lab_mask', type=str, default="./data2/lab_mask/",
                        help="output path ,the jpg save files dir")
    parser.add_argument('--patch_path', type=str, default="./data2/patch_cls/",
                        help="output path ,the jpg save files dir")


    return parser.parse_args()

def load_mapping_path(data_root_dir, iset):
    with open(f'{data_root_dir}/Annotation/segmentation/HE2Json_{iset}.json', 'r',
              encoding='utf-8') as load_f:
        load_values = json.load(load_f)
    return load_values
if __name__ == '__main__':
    args = parse_args()
    data_root_dir = args.data_root_dir
    config_filename = args.config_file  # filelist.json中的json是要处理的svs的名字，如： ['cdh_15_15', 'cdz_2628_2628']

    # tiff_path = os.path.join(data_root_dir, "tif")
    # file_path_json = os.path.join(data_root_dir, "json")
    save_tissue_mask = os.path.join(data_root_dir, "segmentation/tissue_mask")
    save_path_jpg = os.path.join(data_root_dir, "segmentation/img")
    save_lab_mask = os.path.join(data_root_dir, "segmentation/lab_mask")
    patch_path = os.path.join(data_root_dir, "segmentation/patch_cls")

    num_process = args.num_process
    label_list = ['cancer_beside', 'cancer', 'normal_liver']
    # label_list = ['hemorrhage_necrosis', 'other', 'tertiary_lymphatic']

    ##################################### 0. load json data
    HE_info = load_mapping_path(data_root_dir, 'all')
    print(len(HE_info))
    # ##################################### 1.tissue mask
    # make_tissue_mask.make_tissue_mask(data_root_dir, HE_info, save_tissue_mask)
    # ##################################### 2.slide
    # make_slide_cutting.make_slide_cutting(data_root_dir, HE_info, save_path_jpg)
    # ##################################### 3.lab mask
    # make_lab_mask.make_lab_mask(data_root_dir, HE_info, save_path_jpg, save_lab_mask)
    ##################################### 4. creat patch mapping txt file
    # for iset in ['train', 'val', 'test']:
    #     print(f"processing {iset}...")
    #     with open(
    #             f"{data_root_dir}/Annotation/segmentation/HE2Json_{iset}.json"
    #     ) as file_obj:
    #         json_config = json.load(file_obj)
    #     set_files = [name for name, _ in json_config.items()]
    #     create_patch.create_patch(
    #         data_root_dir, HE_info, save_tissue_mask, save_lab_mask,
    #         set_files, label_list, f"{patch_path}/{iset}",
    #         num_process
    #     )
    # # 5. create patch file
    # for iset in ['train', 'val', 'test']:
    #     for label in label_list:
    #         file_path_text = f"{patch_path}/{iset}/{label}.txt"
    #         patch_img_path = f"{data_root_dir}/segmentation/patch_cls/{iset}_img/{label}"
    #         os.makedirs(patch_img_path, exist_ok=True)
    #         create_patch.run(
    #             data_root_dir, HE_info, file_path_text, patch_img_path,
    #             num_process, patch_size=create_patch.patch_size, patch_level=0
    #         )
    ##################################### 5. counting txt file

    for iset in ['test']:
        cancer_count = 0
        no_cancer_count = 0
        for label in ["cancer", "cancer_beside", "hemorrhage_necrosis", "normal_liver", "other", "tertiary_lymphatic"]:
        # for label in ['cancer_beside', 'cancer']:
            type_patch_txt_path = f"{patch_path}/{iset}/{label}.txt"
            f_list = open(type_patch_txt_path)
            lines = f_list.readlines()
            # print(f"The {iset} of {label} is {len(lines)}")
            if label == "cancer":
                cancer_count = len(lines)
            else:
                no_cancer_count += len(lines)
        print(f"The length of cancer is {cancer_count} in {iset}")
        print(f"The length of No cancer is {no_cancer_count} in {iset}")
        print(f"The length of total is {no_cancer_count+cancer_count} in {iset}")
        print("="*20)