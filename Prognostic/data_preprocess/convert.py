import shutil
import os
import pandas as pd
import openslide
from collections import Counter

def _read_magnification(openslide_meta_data):
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

if __name__ == '__main__':
    batch_name = "YouAn"
    xlsx_dir_name = "../mydataset/GT_xlsx/"
    xlsx_all_data = pd.read_excel(os.path.join(xlsx_dir_name, f"{batch_name}.xlsx"))
    xlsx_data = xlsx_all_data[['Names', 'UniqueID', 'RFS_status', 'RFS']]
    xlsx_data_ID = xlsx_data['UniqueID'].values

    raw_dir = "/media/whisper/Elements"
    dst_dir = "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/prognostic/YouAn"

    # slide = openslide.OpenSlide("/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS/prognostic/YouAn/肝移植6+1M23/data.mrxs")
    # print(slide.level_dimensions)

    dimension_details= []
    # for iyouan_data in xlsx_data_ID:
    #     name1, name2 = iyouan_data.split("/")
    #     # new_file_name = iyouan_data.replace("/", "+")
    #     # new_dir = os.path.join(dst_dir, new_file_name)
    #     slide = openslide.OpenSlide(os.path.join(raw_dir, name1, name2+".mrxs"))
    #     mag = _read_magnification(slide.properties)
    #     dimension_details.append(mag)
    #     print(f"{mag}")
    # print(Counter(dimension_details))

    for iyouan_data in xlsx_data_ID:
        print(iyouan_data)
        name1, name2 = iyouan_data.split("/")
        new_file_name = iyouan_data.replace("/", "+")
        new_dir = os.path.join(dst_dir, new_file_name)
        try:
            assert len(os.listdir(new_dir)) == 2, f"miss file in {new_file_name}"

            assert f"{name2}.mrxs" in os.listdir(new_dir)
            sud_data_dir = os.path.join(new_dir, name2)

            raw_sub_data = os.path.join(raw_dir, name1, name2)
            assert len(os.listdir(sud_data_dir)) == len(os.listdir(raw_sub_data)), f"miss file in {name2}"

            slide = openslide.OpenSlide(os.path.join(new_dir, name2 + ".mrxs"))
            mag = _read_magnification(slide.properties)
            dimension_details.append(mag)
            print(f"{mag}")
        except:
            print(f"do {iyouan_data}")
            shutil.copy(
                os.path.join(raw_dir, name1, name2+".mrxs"),
                os.path.join(new_dir, f"{name2}.mrxs"),
            )
            data_dir = os.path.join(new_dir, name2)
            os.makedirs(data_dir, exist_ok=True)
            for idetail in os.listdir(os.path.join(raw_dir, name1, name2)):
                shutil.copy(
                    os.path.join(raw_dir, name1, name2, idetail),
                    os.path.join(data_dir, idetail),
                )
    print(Counter(dimension_details))
        # os.makedirs(new_dir, exist_ok=True)
        # try:
        #     if len(os.listdir(new_dir)) == 1:
        #
        # except Exception as e:
        #     print(f"error when copy {iyouan_data}")
        # break