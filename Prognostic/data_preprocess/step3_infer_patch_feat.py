# conda activate CONCH
import json
from conch.open_clip_custom import create_model_from_pretrained
import torch
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import as_completed



def infer_feat(image_path, preprocess, model, end_str, device):
    emb_name = image_path.replace(".png", end_str)
    print(f"process {image_path}")
    plt_image = Image.open(image_path)
    image = preprocess(plt_image).unsqueeze(0)
    with torch.inference_mode():
        img_emb = model.encode_image(image.to(device), proj_contrast=False, normalize=False)
        np.savez(emb_name, emb=img_emb.cpu().numpy())
    return image_path

def load_undoimg_filelist(json_dir, json_file, aug_method="None"):
    with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as load_f:
        load_values = json.load(load_f)
    img_file_list = []
    for p_idx, isample in enumerate(load_values):
        instance_path = isample["path"]
        win_instance_path = os.path.join(parent_dir, instance_path)
        if aug_method == "None":
            img_dir = f"{win_instance_path}"
        else:
            img_dir = f"{win_instance_path}={aug_method}"

        img_list = [_.split(".png")[0] for _ in os.listdir(img_dir) if _.endswith(".png")]
        emb_list = [_.split(end_str)[0] for _ in os.listdir(img_dir) if _.endswith(end_str)]
        print(f"length of img_list in json {len(img_list)}")
        print(isample["path"])
        img_names = list(set(img_list).difference(set(emb_list)))
        img_path = [os.path.join(img_dir, f"{iimg_name}.png") for iimg_name in img_names]
        img_file_list.extend(img_path)
        # if p_idx == 10:
        #     break

    return img_file_list




if __name__ == '__main__':
    parent_dir = "../dataset/Patch_Images"
    json_dir = "../dataset/RFS_Data_UpLoad"

    multiple_thread = False

    proj_contrast = False
    normalize = False
    end_str = '.npz'


    json_files = [
        'rfs_JiangData32.json',
        'rfs_CY.json',
        'rfs_huashan.json',
        'rfs_huashan2.json',
        'rfs_hz.json',
        'rfs_new_tcga.json',
        'rfs_YouAn.json'
    ]

    method_name_list = [
        "None",
        # "Reinhard",
        # "Ruifrok",
        # "Macenko",
        # "Vahadane"
    ]

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16",
        checkpoint_path="/home/whisper/code/HGTHGT/pretrain/pytorch_model.bin"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)


    all_undo_img_list = []
    for ijson_file in json_files:
        for iaug_method in method_name_list:
            undo_img_list = load_undoimg_filelist(json_dir, ijson_file, iaug_method)
            all_undo_img_list.extend(undo_img_list)

    print(f"undo image length is {len(all_undo_img_list)}")

    flag = 0
    if len(all_undo_img_list) > 0:
        if multiple_thread:
            # print("fuck")
            p = concurrent.futures.ProcessPoolExecutor()
            p.map(infer_feat, all_undo_img_list)
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                jobs, result = [], []
                for i in range(len(all_undo_img_list)):
                    jobs.append(executor.submit(infer_feat, all_undo_img_list[i], preprocess, model, end_str, device))
                for job in as_completed(jobs):
                    result.append(job.result())
        else:
            for i, img_path in tqdm(enumerate(all_undo_img_list)):
                infer_feat(img_path, preprocess, model, end_str, device)
                if flag % 100 == 0:
                    print("\n++++++++++++++++++++++++++"+str(len(all_undo_img_list)-flag)+"++++++++++++++++++++++++++\n")
                    flag += 1
