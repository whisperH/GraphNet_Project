import os
import json
import numpy as np
from tqdm import tqdm
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
from matplotlib import pyplot as plt
# exp_name = "res34_69tif_1cls_v2.1.0_ckppt20_thre05_process_infer_v2"
exp_name = "20221012_3tif_768_all_cls_tissue06_lab08_new_uniq_v1"


root = "/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/res34_69tif_1cls_v2.1.0/process_res/infer_v2_test_result_ckpt20_thresh05.json"
file0 = open(root,'r',encoding="utf-8")
lines0 = file0.readlines()
file0.close()

grid_size = 256
count = 0

save_path = os.path.join("/home/data/huangxiaoshuang/vis", exp_name, "vis_res_prob_v2")
if not os.path.exists(save_path):
    os.system("mkdir -p " + save_path)


data_root = "/home/data2/medicine/20221012_3tif_768_all_cls_tissue06_lab08/"
draw_dic = {} # tif_name:cv_img

# 汇总每张tif的lab_mask图像
for i in (os.listdir(data_root+"img")):
    k = i.split("_")
    print(os.path.join(data_root, "lab_mask", "cancer_beside", k[0] + "_mask_64.png"))
    tt = cv2.imread(os.path.join(data_root, "lab_mask", "cancer", k[0] + "_mask_64.png"))
    h,w,c = tt.shape
    tmp_img = np.zeros((h, w, c), dtype=int)
    # draw_dic[k[0]] = np.zeros((h, w), dtype=np.float16)
    draw_dic[k[0]] = np.zeros((h, w))
    print("{} draw numpy size: {}, {}".format(k[0], h, w))
    try:
        for j in os.listdir(data_root + "lab_mask"):
            if j != "cancer":
                # print(os.path.join(data_root, "lab_mask", j, k[0] + "_mask_64.png"))
                tmp_img2 = cv2.imread(os.path.join(data_root, "lab_mask", j, k[0] + "_mask_64.png"))
                tmp_img = tmp_img + tmp_img2
    except:
        pass
    cv2.imwrite(os.path.join(data_root, k[0] + "_no_cancer.png"), tmp_img)
    
all_grid_dic = {}
for line in lines0:
    dic = json.loads(line)
    tif_name = dic["filename"].split("_")[0]
    left_top_x = dic["x_center"] - 384
    left_top_y = dic["y_center"] - 384
    grid_center_x = dic["grid_idx"]//3*256 + left_top_x + grid_size // 2
    grid_center_y = dic["grid_idx"]%3*256 + left_top_y + grid_size // 2
    dic["grid_center_x"] = grid_center_x
    dic["grid_center_y"] = grid_center_y
    
    if tif_name not in all_grid_dic.keys():
        all_grid_dic[tif_name]=[dic]
    else:
        all_grid_dic[tif_name].append(dic)

for tif, grid_lst in draw_dic.items():
    w,h = draw_dic[tif].shape
    # plt.figure(figsize=(w/100,h/100), frameon=False)
    # plt.figure(frameon=False)
    for grid in tqdm(all_grid_dic[tif]):
        for y in range((grid["grid_center_y"]-128)//64, (grid["grid_center_y"]+128)//64):
            for x in range((grid["grid_center_x"]-128)//64, (grid["grid_center_x"]+128)//64):
                # draw_dic[tif][y][x] = grid["pred_logit"]
                draw_dic[tif][y][x] = grid["label"]
                
    plt.imshow(draw_dic[tif], vmin=0, vmax=1, cmap='gray',interpolation=None,filternorm=False,filterrad=1)
    # plt.imshow(draw_dic[tif], vmin=0, vmax=1, cmap='seismic',interpolation=None,filternorm=False,filterrad=1)
    
    # plt.imshow(draw_dic[tif], vmin=0, vmax=1, cmap='copper',interpolation=None,filternorm=False,filterrad=1)
    cv2.imwrite(save_path+'/{}_cv2.png'.format(tif), draw_dic[tif]*255)
    plt.colorbar()
    f = plt.gcf()  #获取当前图像
    f.savefig(save_path+'/{}.png'.format(tif))
    f.clear()  #释放内存

