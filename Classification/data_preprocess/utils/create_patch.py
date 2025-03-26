import os
import time
import sys
from multiprocessing import Pool
from openslide import OpenSlide
import numpy as np
from skimage.transform.integral import integral_image, integrate
import cv2
from shutil import copyfile
import math
import json
# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

# 将坐标存储在对应的txt文件中



# patch的大小为768*768
patch_size = 768
patch_size_lv = patch_size #代码中并没有实际试用到patch_size_lv
patch_level = 0 #未用到的参数
mask_downsample = 64 #缩放倍数，这个参数和生成略所图的参数一致，根据生成略缩图的参数进行设置
stride = patch_size #步长，也就是patch每次移动的大小，一般跟patch_size一致即可


def create_txt_sequence(data_root_dir,file_path_tis_mask,file_path_mask, HE_info, list,\
                     file_path_txt,kind,patch_size,mask_downsample,stride):

    txt = open(file_path_txt,'w+')
    for file_name, file_info in HE_info.items():

    # for file in os.listdir(file_path_json):
    #     if file[-3:] == 'txt':
    #         continue
    #     file_name=file[:-5]
    #     if file_name[-10:] == '_tif_Label':
    #         file_name = file_name[:-10]

        # 这里改了一下，因为原来是读json名称
        lab_mask_name = os.path.join(file_path_mask , file_name + '_mask_' + str(mask_downsample) + '.png')
        print(lab_mask_name)
        if (file_name in list) and (os.path.exists(lab_mask_name)):
            file_name = file_name.split('_tif_Label')[0]
            tissue_mask_name = os.path.join(file_path_tis_mask, file_name + '_tissue_mask_' + str(mask_downsample) + '.png')
            print(tissue_mask_name)
            tissue_mask = cv2.imread(tissue_mask_name, 0)
            integral_image_tissue = integral_image(tissue_mask.T / 255)
            # 生成切片的积分图像
            # 一个图像内矩形区域的积分是指这个矩形区域内所有灰度值的和

            lab_mask = cv2.imread(lab_mask_name, 0)
            integral_image_lab = integral_image(lab_mask.T / 255)
            slide = OpenSlide(os.path.join(data_root_dir, file_info['HE_filepath']))
            slide_w_lv_0, slide_h_lv_0 = slide.dimensions
            slide_w_downsample = slide_w_lv_0 / mask_downsample
            slide_h_downsample = slide_h_lv_0 / mask_downsample
            # 768/64
            size_patch_lv_k = int(patch_size / mask_downsample)  # patch在第mask_level层上映射的大小

            # 建立一个树形结构的轮廓，仅保存4点信息
            contours_lab, _ = cv2.findContours(lab_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

            # if len(p_left) and len(p_right) and len(p_top) and len(p_bottom) != 0:
            #     p_x_left = min(p_left)
            #     p_x_right = max(p_right)
            #     p_y_top = min(p_top)
            #     p_y_bottom = max(p_bottom)

            # 1024/64
            stride_lv=int(stride/mask_downsample)
            print(stride_lv)
            list_type = []

            for contour_idx in range(len(contours_lab)):
                p_x_left = p_left[contour_idx]
                p_x_right = p_right[contour_idx]
                p_y_top = p_top[contour_idx]
                p_y_bottom = p_bottom[contour_idx]
                for x in range(p_x_left, p_x_right, stride_lv):
                    for y in range(p_y_top, p_y_bottom, stride_lv):
                        x_lv = int(x + size_patch_lv_k / 2)
                        y_lv = int(y + size_patch_lv_k / 2)
                        if (y + size_patch_lv_k> slide_h_downsample) or \
                                (x + size_patch_lv_k > slide_w_downsample):
                            continue
                        # 求解积分
                        tissue_integral = integrate(integral_image_tissue, \
                                                    (x, y), \
                                                    (x + size_patch_lv_k - 1,y + size_patch_lv_k - 1))
                        tissue_ratio = tissue_integral / (size_patch_lv_k ** 2)
                        lab_integral = integrate(integral_image_lab, \
                                                 (x, y), \
                                                 (x + size_patch_lv_k - 1,
                                                  y + size_patch_lv_k - 1)
                                                 )
                        lab_ratio = lab_integral / (size_patch_lv_k ** 2)
                        # if tissue_ratio > 0.6 or lab_ratio > 0.8:
                        #     list_type.append([x, y])
                        # else:
                        #     continue
                        if tissue_ratio < 0.7 or lab_ratio < 0.5:
                            continue
                        list_type.append([x, y])


            for i, item in enumerate(list_type):
                x = item[0]
                y = item[1]


                patch_x_lv_0 = str((round(int(x + size_patch_lv_k / 2) * mask_downsample)))
                patch_y_lv_0 = str((round(int(y + size_patch_lv_k / 2) * mask_downsample)))
                print(file_name+" "+ patch_x_lv_0+" "+patch_y_lv_0+kind)
                imgname = file_name + "_" + patch_x_lv_0 + '_' + patch_y_lv_0 + '.png'
                #########on-off###########
                # with open("/home/data/huangxiaoshuang/json_files/20221012_3tif_tissue06_lab08_tmp.json", "a") as f:
                # with open("/home/data/huangxiaoshuang/json_files/20221031_69tif_tissue07_lab05_stride2.json", "a") as f:
                #     tmp_dict = {
                #         "name":file_name,
                #         "x_center":int(patch_x_lv_0),
                #         "y_center":int(patch_y_lv_0),
                #         "class":kind,
                #         "patch_size":patch_size,
                #         "level":patch_level
                #     }
                #     tmp_dict.update(file_info)
                #     f.write(json.dumps(tmp_dict)+"\n")
                txt.writelines([imgname, ',', file_name, ',', patch_x_lv_0, ',', patch_y_lv_0, ',', kind, '\n'])


    txt.close()


# 生成分块
def process(opts):
    i, imgname, pid, x_center, y_center, HE_file_path, path_patch, patch_size, patch_level = opts
    x = int(int(float(x_center)) - patch_size / 2)
    y = int(int(float(y_center)) - patch_size / 2)
    slide = OpenSlide(HE_file_path)
    img = slide.read_region(
        (x, y), 0,
        (patch_size, patch_size)).convert('RGB')
    if not os.path.exists(os.path.join(path_patch, imgname)):

        print(os.path.join(path_patch, imgname))
        img.save(os.path.join(path_patch, imgname))
    slide.close()



def remove_duplicates(origin_pth, aim_pth):
    f_read=open(origin_pth+'.txt','r',encoding='utf-8')     #将需要去除重复值的txt文本重命名text.txt
    f_write=open(aim_pth+'.txt','w',encoding='utf-8')  #去除重复值之后，生成新的txt文本 --“去除重复值后的文本.txt”
    data=set()
    for a in [a.strip('\n') for a in list(f_read)]:
        if a not in data:
            f_write.write(a+'\n')
            data.add(a)
    f_read.close()
    f_write.close()
    print('完成')



# 多进程处理
def run(
        data_root_dir, HE_info, file_path_text, patch_img_path,
        num_process,patch_size,patch_level
):
    already_done = []
    for ialready_done_sample in os.listdir(patch_img_path):
        ialready_done_sample_dir = os.path.join(patch_img_path, ialready_done_sample)
        for ialready_done_patch_img_name in os.listdir(ialready_done_sample_dir):
            already_done.append(ialready_done_patch_img_name)
    opts_list = []
    list = []
    infile = open(file_path_text, "r")
    for i, line in enumerate(infile):
        # print(line)
        # print(line.strip('\n').split(','))
        # pid: filename, 存储在imgname 中
        imgname, pid, x_center, y_center, _ = line.strip('\n').split(',')
        if imgname in already_done:
            print(f"{imgname} already done!!!")
            continue
        else:
            list.append([imgname, pid, x_center, y_center])
    count = len(list)
    print(f"the counts of patch list: {count} in {file_path_text}")
    infile.close()
    for i in range(count):
        imgname = list[i][0]
        pid = list[i][1]
        x_center = list[i][2]
        y_center = list[i][3]
        HE_file_path = os.path.join(data_root_dir, HE_info[pid]['HE_filepath'])
        path_patch = os.path.join(patch_img_path, str(pid))
        os.makedirs(path_patch, exist_ok=True)

        opts_list.append((i, imgname, pid, x_center, y_center, HE_file_path, path_patch, patch_size, patch_level))
    pool = Pool(processes=num_process)
    pool.map(process, opts_list)


def create_patch(
        data_root_dir, HE_info, file_path_tis_mask, file_path_lab_masks,
        file_id_list, label_list, patch_path, num_process = 5
):
    """
    file_path_tif, file_path_json, file_path_tis_mask, file_path_lab_mask已经生成的文件
    file_id_list是要生成patch的文件id，
    如 file_id_list = ['bmq_305_745',
                  'cby_350_350',
                  'cdh_15_15',
                  'cdz_2628_2628']
    label_list 是要处理的label，如label_list =  ['cancer_beside', 'cancer', 'normal_liver']
    patch_path 要存放patch的路径，注意最后一层往往是train或者valid如 patch_path = "./data2/patch_cls/train/"
    """
    file_path_texts = patch_path
    os.makedirs(patch_path, exist_ok=True)
    print('Creating txt!')
    time_now = time.time()
    # tumor_beside_mask,train,valid
    # label_list = ['cancer_beside', 'cancer', 'normal_liver']
    for label in label_list:
        file_path_lab_mask = os.path.join(file_path_lab_masks, label)
        file_path_text = os.path.join(file_path_texts, label + ".txt")
        create_txt_sequence(data_root_dir, file_path_tis_mask, file_path_lab_mask, HE_info, file_id_list, \
                            file_path_text, label, patch_size, mask_downsample, stride)
    time_spent = (time.time() - time_now) / 60
    print('Creating txt for %f min!' % time_spent)

    print('Making patch!')
    time_now = time.time()
    #####on-off#########
    # for label in label_list:
    #     file_path_text = file_path_texts + "/" + label + ".txt"
    #     run(file_path_tif, file_path_text, patch_path, num_process, patch_size, patch_level)
    time_spent = (time.time() - time_now) / 60
    print('Making patch  for %f min!' % time_spent)


# if __name__ == '__main__':
#     #in
#
#     data_root_dir = "/media/whisper/Newsmy/WSI_RawData/CY_SL_FD_HS"
#     file_path_tis_mask = os.path.join(data_root_dir, "segmentation/tissue_mask")
#     file_path_lab_masks = os.path.join(data_root_dir, "segmentation/lab_mask")
#     patch_path = os.path.join(data_root_dir, "segmentation/patch_cls/train")
#     HE_info = {
#         "9123526T 杨邦宏 T HE": {
#             "HE_filepath": "segmentation/WSI_Data/9123526T 杨邦宏 T HE.tif",
#             "file_suffix": ".tif",
#             "seg_filepath": "Annotation/segmentation/GT/9123526T 杨邦宏 T HE_tif_Label.json"
#         },
#         "9187359 康涛 癌 HE": {
#             "HE_filepath": "segmentation/WSI_Data/9187359 康涛 癌 HE.tif",
#             "file_suffix": ".tif",
#             "seg_filepath": "Annotation/segmentation/GT/9187359 康涛 癌 HE_tif_Label.json"
#         }
#     }
#
#     #out
#     file_id_list = ['9123526T 杨邦宏 T HE', '9187359 康涛 癌 HE']
#
#     label_list = ['cancer_beside', 'cancer', 'normal_liver']
#     #patch path
#     # create_patch(file_path_tif, file_path_json, file_path_tis_mask, file_path_lab_masks, file_id_list, label_list, patch_path)
#     create_patch(
#         data_root_dir, HE_info, file_path_tis_mask, file_path_lab_masks,
#         file_id_list, label_list, patch_path
#     )

