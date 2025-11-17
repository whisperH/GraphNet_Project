# conda activate CONCH
import json
import argparse
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor,as_completed, wait, ALL_COMPLETED
import shutil
import os
import numpy as np
from functools import partial

from PIL import Image

from CONCH.conch.open_clip_custom import create_model_from_pretrained
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="CONCH Infer Script")
    #config
    # parser.add_argument('--data_root_dir', type=str, default="/media/server/Newsmy/WSI_RawData/CY_SL_FD_HS/prognostic")
    parser.add_argument('--data_root_dir', type=str, default="/media/server/Newsmy/WSI_RawData/CY_SL_FD_HS/MoleculeValidation")
    parser.add_argument('--parent_dir', type=str, default="/media/server/Newsmy/WSI_PatchData/CY_SL_FD_HS/250715_res34_upload")
    parser.add_argument('--patch_embedding_dir', type=str, default="/media/server/Newsmy/WSI_PatchData/HCC/CONCH_Coord")
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_process', type=int, default=1)

    return parser.parse_args()

def get_done_list(done_center_dir):
    already_done = []
    for ialready_done_sample_name in os.listdir(done_center_dir):
        done_sample_dir = os.path.join(done_center_dir, ialready_done_sample_name)
        if os.path.exists(done_sample_dir):
            for ialready_done_patch_img_name in os.listdir(done_sample_dir):
                already_done.append(f"{ialready_done_patch_img_name.split('.npz')[0]}")
        else:
            os.makedirs(done_sample_dir, exist_ok=True)
    return already_done

def load_patch_coord_from_file(cord_filepath, already_done, embedding_filepath):
    with open(cord_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    process_list = {}
    for iHE_path, HE_info in data.items():
        if HE_info['PID'] in already_done:
            print(f"{HE_info['PID']} already done!!!")
            continue
        process_list[iHE_path] = []
        HE_info['path'] = os.path.join(embedding_filepath, HE_info['path'])
        process_list[iHE_path] = HE_info
    return process_list


def load_patch_coord_from_folder(img_filefolder, dataset, already_done, embedding_filepath):
    samples_list = os.listdir(img_filefolder)
    process_list = []
    for sample_id in samples_list:
        embed_sample_dir = os.path.join(embedding_filepath, dataset, sample_id)
        os.makedirs(embed_sample_dir, exist_ok=True)
        for ipatch in os.listdir(os.path.join(img_filefolder, sample_id)):
            if ipatch.split(".png")[0] in already_done:
                print(f"{ipatch} of {sample_id} already done!!!")
                continue
            embed_name = ipatch.split('.png')[0]
            # _sample_id, _x, _y, _conf = embed_name.split('_')
            # assert _sample_id == sample_id
            process_list.append({
                "img_path": os.path.join(img_filefolder, sample_id, ipatch),
                "patch_embedding_filepath": os.path.join(embedding_filepath, dataset, sample_id, f"{embed_name}.npz"),
            })
    return process_list

class Conch_MIL_fc(nn.Module):
    def __init__(self, size_arg="small", dropout=0., n_classes=2, ckpt_path="",
                 embed_dim=1024):
        super().__init__()
        self.conch, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path="./CONCH/checkpoints/conch/pytorch_model.bin",
            force_image_size=(256, 256)
        )
        self.conch.forward = partial(self.conch.encode_image, proj_contrast=False, normalize=False)

    def forward(self, h, return_features=False):
        feature_h = self.conch(h)
        # h = self.fc(feature_h.float())
        # logits = self.classifier(feature_h.float())  # K x 2

        # y_probs = F.softmax(logits, dim=1)
        # return logits, y_probs, feature_h
        return feature_h
def initiate_model(args, device='cuda'):
    print('Init Model')
    model = Conch_MIL_fc()
    model.to(device)
    return model



def load_undoimg_filelist(json_dir, json_file, aug_method="None"):
    with open(os.path.join(json_dir, json_file), 'r', encoding='utf-8') as load_f:
        load_values = json.load(load_f)
    img_file_list = []
    for p_idx, isample in enumerate(load_values):
        instance_path = isample["path"]
        win_instance_path = os.path.join(args.parent_dir, instance_path)
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
        # if p_idx == 1:
        #     break

    return img_file_list


class MultiInferDataset(Dataset):
    def __init__(self, all_undo_img_list, patch_size=256, preprocess=None):
        self.all_undo_img_list = all_undo_img_list
        self.preprocess = preprocess

    def __len__(self):
        return len(self.all_undo_img_list)

    def __getitem__(self, idx):
        # 这里实现你的图片加载和预处理
        img = Image.open(self.all_undo_img_list[idx]['img_path'])
        saved_filepath = self.all_undo_img_list[idx]['patch_embedding_filepath']
        # Sample_id = self.all_undo_img_list[idx]['Sample_id']
        # center_x, center_y = self.all_undo_img_list[idx]['coord']
        # emb_name = f"{Sample_id}_{center_x}_{center_y}"
        # saved_filepath = os.path.join(self.all_undo_img_list[idx]['patch_embedding_dir'], f"{emb_name}.npz")
        return self.preprocess(img), saved_filepath



def multiprocess_save(process_list):
    features, saved_path = process_list
    np.savez_compressed(saved_path, emb=features.cpu().numpy())
    return 1


def copy_png_files(source_folder, destination_folder):
    """
    将源文件夹中所有以.png结尾的文件复制到目标文件夹

    参数:
    source_folder (str): 源文件夹路径
    destination_folder (str): 目标文件夹路径
    """
    # 确保目标文件夹存在
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"创建目标文件夹: {destination_folder}")

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件是否以.png结尾
        if filename.lower().endswith('.png'):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            # 复制文件
            shutil.copy2(source_path, destination_path)
            print(f"已复制: {filename}")

    print("复制操作完成!")

if __name__ == '__main__':
    args = parse_args()
    data_root_dir = args.data_root_dir
    parent_dir = args.parent_dir
    patch_embedding_dir = args.patch_embedding_dir
    batch_size = args.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    multiple_thread = False
    end_str = '.npz'

    model = initiate_model(args, device)

    # for dataset in ['huashan2', 'huashan', 'YouAn', 'hz', 'CY', "JiangData"]:
    for dataset in ['ZY4HE26605ZHL', 'file_202051101']:
        print("Loading already done data")
        done_center_dir = os.path.join(patch_embedding_dir, dataset)
        os.makedirs(done_center_dir, exist_ok=True)
        already_done = get_done_list(done_center_dir)

        print("Loading undo data")
        img_filefolder = os.path.join(parent_dir, f"{dataset}")
        all_undo_img_list = load_patch_coord_from_folder(img_filefolder, dataset, already_done, patch_embedding_dir)

        print("F***k")
        if len(all_undo_img_list) > 0:
            iHE_dataset = MultiInferDataset(
                all_undo_img_list,
                preprocess=model.preprocess
            )
            loader = DataLoader(
                iHE_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # 预处理进程数
                # collate_fn=inference_collate,
                pin_memory=True  # 加速CPU到GPU传输
            )
            with torch.inference_mode():
                for batch, saved_filepath in tqdm(loader):
                    batch = batch.cuda(non_blocking=True)
                    feature_h = model(batch)

                    split_tensors = torch.split(feature_h, 1, dim=0)
                    for iprocess in list(zip(split_tensors, saved_filepath)):
                        multiprocess_save(iprocess)
                    # with ThreadPoolExecutor(max_workers=2) as pool:
                    #     all_task = []
                    #     for iprocess in list(zip(split_tensors, saved_filepath)):
                    #         all_task.append(
                    #             pool.submit(multiprocess_save, iprocess)
                    #         )
                    #     wait(all_task, return_when=ALL_COMPLETED)
                    #     print("----complete of batch-----")

        PrognosticInfo = []
        # generate the Prognostic information
        samples_list = os.listdir(img_filefolder)
        for isample_id in samples_list:
            PrognosticInfo.append({
                "RFS_daytime": -1,
                "RFS_time": -1,
                "events": -1,
                "PID": isample_id,
                "path": f"{dataset}/{isample_id}",
                "name": isample_id,
            })
        with open(f"/media/server/Newsmy/WSI_PatchData/HCC/CONCH_Coord/rfs_{dataset}.json", "w+", encoding="utf-8") as load_f:
            json.dump(PrognosticInfo, load_f, ensure_ascii=False, indent=4)