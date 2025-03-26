import os
import openslide
import json
import shutil
from sklearn.model_selection import train_test_split

from collections import Counter
def check_data_repeat():
    seg_anno_train_filelist = os.listdir("/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Annotation/segmentation/train")
    seg_anno_test_filelist = os.listdir("/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Annotation/segmentation/test")
    seg_anno_val_filelist = os.listdir("/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Annotation/segmentation/val")

    train_list = set(seg_anno_train_filelist)
    assert len(train_list) == len(seg_anno_train_filelist)
    val_list = set(seg_anno_val_filelist)
    assert len(val_list) == len(seg_anno_val_filelist)
    test_list = set(seg_anno_test_filelist)
    assert len(test_list) == len(seg_anno_test_filelist)

    print(set(train_list) & set(val_list))
    print(set(train_list) & set(test_list))
    print(set(val_list) & set(test_list))

def read_magnification(file_path):
    slide = openslide.OpenSlide(file_path)
    openslide_meta_data = slide.properties
    mag_obj = -1

    if 'openslide.objective-power' in openslide_meta_data:
        mag_obj = int(openslide_meta_data['openslide.objective-power'])

    mag_mmp = -1
    if 'openslide.mpp-x' in openslide_meta_data:
        mmp = float(openslide_meta_data['openslide.mpp-x'])
        if abs(mmp - 0.25) < 0.05:
            mag_mmp = 40
        elif abs(mmp - 0.5) < 0.05:
            mag_mmp = 20
        elif abs(mmp - 1.0) < 0.05:
            mag_mmp = 10
        elif abs(mmp - 2.0) < 0.05:
            mag_mmp = 5

    # use mag_obj at first
    if mag_obj in [5, 10, 20, 40]:
        return mag_obj

    if mag_mmp in [5, 10, 20, 40]:
        return mag_mmp

    return -1

def get_name(HE_list):
    name_list = []
    suffix_list = []
    for iHE in HE_list:
        if '.svs' in iHE:
            name_list.append(iHE.split('.svs')[0])
            suffix_list.append('.svs')
        elif '.tif' in iHE:
            name_list.append(iHE.split('.tif')[0])
            suffix_list.append('.tif')
    return name_list, suffix_list


# step1: move Segmentation train set to Train Dir
def move_HE_to_train():
    seg_anno_filelist = os.listdir("/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Annotation/segmentation/GT")

    dst_dir = "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/segmentation/WSI_Data"

    huashan2bak_dir = "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Unknown/huashan2bak"
    hzbak_dir = "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Unknown/hz_bak"
    huashan_bak_dir = "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Unknown/huashan_bak"
    other_dir = "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Unknown/other"

    icheck_dirs = [
        huashan2bak_dir, hzbak_dir, huashan_bak_dir, other_dir
    ]

    for iseg_anno_file in seg_anno_filelist:
        HE_name = iseg_anno_file.split("_tif_Label")[0]
        for icheck_dir in icheck_dirs:
            HE_list_Unknown, suffix_list = get_name(
                os.listdir(icheck_dir)
            )
            if HE_name in HE_list_Unknown:
                idx = HE_list_Unknown.index(HE_name)
                try:
                    shutil.move(
                        os.path.join(icheck_dir, HE_name + f"{suffix_list[idx]}"),
                        os.path.join(dst_dir, HE_name + f"{suffix_list[idx]}")
                    )
                except:
                    print(f"failed {HE_name} in {icheck_dir}")


# step2: generate mapping of HE and annotation
def HE_mapping():
    seg_anno_filelist = os.listdir("/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Annotation/segmentation/GT")
    root_dir = '/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS'

    HE_list_huashan = "prognostic/huashan"
    HE_list_CY = "prognostic/CY"
    HE_list_hz = "prognostic/hz"
    other_dir = "segmentation/WSI_Data"

    HE_Anno_mapping_file = {}

    icheck_dirs = [
        HE_list_huashan, HE_list_CY, HE_list_hz, other_dir
    ]
    mag_count = []
    for iseg_anno_file in seg_anno_filelist:
        HE_name = iseg_anno_file.split("_tif_Label")[0]

        flag = 0
        for icheck_dir in icheck_dirs:
            HE_list_Unknown, suffix_list = get_name(
                os.listdir(os.path.join(root_dir, icheck_dir))
            )
            if HE_name in HE_list_Unknown:
                idx = HE_list_Unknown.index(HE_name)
                HE_suffix = suffix_list[idx]
                HE_Anno_mapping_file[HE_name] = {
                    "HE_filepath": f"{icheck_dir}/{HE_name}{suffix_list[idx]}",
                    "file_suffix": HE_suffix,
                    "seg_filepath": f"Annotation/segmentation/GT/{iseg_anno_file}",
                }

                # try:
                #     mag = read_magnification(os.path.join(
                #         root_dir, icheck_dir, HE_name + f"{suffix_list[idx]}"
                #     ))
                #     print(f"{HE_name}: {mag}")
                # except Exception as e:
                #     print(f"fuck processing {icheck_dir}/{HE_name}{suffix_list[idx]}")
                #     exit(-1)
                # mag_count.append(mag)
                flag = 1
                continue
        if flag == 0:
            print(f"{HE_name} not found")
    with open(f'/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Annotation/segmentation/HE2Json_all.json',
              'w') as f:
        json.dump(HE_Anno_mapping_file, f, ensure_ascii=False)

    all_idx = [name for name, _ in HE_Anno_mapping_file.items()]
    X_test = ["D17-02013-01", "D18-02107-10", "D18-02242-04"]
    train_val_idx = list(set(all_idx) - set(X_test))

    X_train, X_val, y_train, y_val = train_test_split(
        # HE_Anno_mapping_file,
        train_val_idx,
        [_ for _ in range(len(train_val_idx))],
        test_size=1 - 0.8, shuffle=True, random_state=1
    )
    # 205 59 29 = 293
    print(f"X_train: {len(X_train)}")
    print(X_train)
    print(f"X_val: {len(X_val)}")
    print(X_val)
    print(f"X_test: {len(X_test)}")
    print(X_test)

    set_info = {
        "train": X_train,
        "val": X_val,
        "test": X_test,
    }
    for iset_name, iset in set_info.items():
        dict_list = {}
        for iname in iset:
            dict_list[iname] = HE_Anno_mapping_file[iname]
        with open(f'/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Annotation/segmentation/HE2Json_{iset_name}.json', 'w') as f:
            json.dump(dict_list, f, ensure_ascii=False)

if __name__ == '__main__':
    seg_anno_filelist = os.listdir("/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/Annotation/segmentation/GT")
    HE_mapping()

