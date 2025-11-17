import os
import json
import numpy as np
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
exp_name = "1cls_v2.4.3_ckpt99_thresh06_new"
# exp_name = "1018_res34_15tif_v2_debug"

root = "/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/1cls_v2.4.3/test_result.json"
# root = "/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/1018_res34_15tif_v2/test_result_latest_tissue06_lab08_debug.json"
file0 = open(root,'r',encoding="utf-8")
lines0 = file0.readlines()
file0.close()

count = 0

save_path = os.path.join("/home/data/huangxiaoshuang/vis", exp_name, "draw_hard_case")
if not os.path.exists(save_path):
    os.system("mkdir -p " + save_path)


data_root = "/home/data2/medicine/20221012_3tif_768_all_cls_tissue06_lab08/"
draw_dic = {}

# 汇总每张tif的lab_mask图像
for i in (os.listdir(data_root+"img")):
    k = i.split("_")
    # 读取每张tif缩略图
    draw_dic[k[0]] = cv2.imread(data_root+"img/"+i)
    print(os.path.join(data_root, "lab_mask", "cancer_beside", k[0] + "_mask_64.png"))
    tt = cv2.imread(os.path.join(data_root, "lab_mask", "cancer", k[0] + "_mask_64.png"))
    h,w,c = tt.shape
    tmp_img = np.zeros((h, w, c), dtype=int)
    try:
        for j in os.listdir(data_root + "lab_mask"):
            if j != "cancer":
                # print(os.path.join(data_root, "lab_mask", j, k[0] + "_mask_64.png"))
                tmp_img2 = cv2.imread(os.path.join(data_root, "lab_mask", j, k[0] + "_mask_64.png"))
                tmp_img = tmp_img + tmp_img2
    except:
        pass
    cv2.imwrite(os.path.join(data_root, k[0] + "_no_cancer.png"), tmp_img)

t =0 
count_dic = {"TP":0, "FP":0, "TN":0, "FN":0}
color_lst = {"TP":(15, 94, 56),"FP":(255, 255, 0),"TN":(0, 255, 0),"FN":(255, 0, 0)}
# color_lst = {"FN":(15, 94, 56),"FP":(255, 255, 0),"TN":(0, 255, 0),"TP":(255, 0, 0)}
for idx, line in enumerate(lines0):
    res_dic = json.loads(line)
    left_top_x = res_dic["x_center"] - 384
    left_top_y = res_dic["y_center"] - 384
    grid_left_top_x = (res_dic["grid_idx"]//3*256 + left_top_x) // 64
    grid_left_top_y = (res_dic["grid_idx"]%3*256 + left_top_y) // 64
    # draw hard case
    if res_dic["is_correct"]:
        if res_dic["label"]==1: # TP暗绿色bgr,rgb
            # cv2.rectangle(draw_dic[res_dic["filename"].split("_")[0]], ((res_dic["x_center"]-128)//64, (res_dic["y_center"]-128)//64), ((res_dic["x_center"]+128)//64, (res_dic["y_center"]+128)//64), (128, 20, 48), 1)
            cv2.rectangle(draw_dic[res_dic["filename"].split("_")[0]], (grid_left_top_x, grid_left_top_y), (grid_left_top_x+4, grid_left_top_y+4), color_lst["TP"], 1)
            count_dic["TP"] += 1
        if res_dic["label"]==0: # TN
            cv2.rectangle(draw_dic[res_dic["filename"].split("_")[0]], (grid_left_top_x, grid_left_top_y), (grid_left_top_x+4, grid_left_top_y+4), color_lst["TN"], 1)
            count_dic["TN"] += 1
    else:
        if res_dic["label"]==0: # FP
            cv2.rectangle(draw_dic[res_dic["filename"].split("_")[0]], (grid_left_top_x, grid_left_top_y), (grid_left_top_x+4, grid_left_top_y+4), color_lst["FP"], 1)
            count_dic["FP"] += 1
            if "D18-02107-10" in res_dic["filename"]:
                t += 1
                # print(t,line)
        if res_dic["label"]==1: # FN
            cv2.rectangle(draw_dic[res_dic["filename"].split("_")[0]], (grid_left_top_x, grid_left_top_y), (grid_left_top_x+4, grid_left_top_y+4), color_lst["FN"], 1)
            count_dic["FN"] += 1

for k in draw_dic.keys():
    h,w,c = draw_dic[k].shape
    idx = 1
    for kk,v in color_lst.items():
        # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
        imgzi = cv2.putText(draw_dic[k], kk, (w-60, h-40*idx), font, 1, v, 1)
        idx += 1
    cv2.imwrite(os.path.join(save_path, k+".png"), imgzi)

print(count_dic)