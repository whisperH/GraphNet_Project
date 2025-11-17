import os
import json
import numpy as np
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
import time
from tqdm import tqdm

start_time = time.time()


# build new res json path
save_root = os.path.join("/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/res34_69tif_1cls_v2.1.0/", "process_res")
save_json_name = "test_result_ckpt20_thresh05.json"
os.makedirs(save_root, exist_ok=True)

# search radius ratio
radius = 1.5
# read infer res json
source_res_json_path = "/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/res34_69tif_1cls_v2.1.0/test_result_ckpt20_thresh05.json"
source_res_json = open(source_res_json_path, 'r', encoding="utf-8")
source_res_json_lines = source_res_json.readlines()
source_res_json.close()

count = 0
grid_size = 256

new_res_json = open(os.path.join(save_root, save_json_name), 'w', encoding="utf-8")

time0 = time.time()
# get all hard grid
hard_grid_list = []
all_grid_dic = {}
for line in source_res_json_lines:
    dic = json.loads(line)
    if not dic['is_correct']:
        hard_grid_list.append(dic)
    tif_name = dic["filename"].split("_")[0]
    if tif_name not in all_grid_dic.keys():
        all_grid_dic[tif_name]=[dic]
    else:
        all_grid_dic[tif_name].append(dic)

print("hard_grid_list: ", len(hard_grid_list))
print("get all hard grid: {} s".format(time.time()-time0))

print(len(all_grid_dic))
# get hard case and its neighbor
time1 = time.time()
hard_and_neighbor = []
bar = tqdm(hard_grid_list)
for hard_case in bar:
    left_top_x = hard_case["x_center"] - 384
    left_top_y = hard_case["y_center"] - 384
    grid_center_x = hard_case["grid_idx"]//3*256 + left_top_x + grid_size // 2
    grid_center_y = hard_case["grid_idx"]%3*256 + left_top_y + grid_size // 2
    neighbor = []
    # print(len(all_grid_dic[hard_case["filename"]]), "----")
    for dic in all_grid_dic[dic["filename"].split("_")[0]]:
        left_top_x = dic["x_center"] - 384
        left_top_y = dic["y_center"] - 384
        neighber_grid_center_x = dic["grid_idx"]//3*256 + left_top_x + grid_size // 2
        neighber_grid_center_y = dic["grid_idx"]%3*256 + left_top_y + grid_size // 2

        if abs(neighber_grid_center_x-grid_center_x)<(grid_size*radius) and \
            abs(neighber_grid_center_y-grid_center_y)<(grid_size*radius):
            neighbor.append(dic)
    # whether hard_case has 9 neighbors
    if len(neighbor) > 7:
        neighbor.append(hard_case)
        hard_and_neighbor.append(neighbor)

print("hard_and_neighbor: ", len(hard_and_neighbor))
print("get hard case and its neighbor: {} s".format(time.time()-time1))

# change hardcase class
time2 = time.time()
changed_hard_case = {}
for item in hard_and_neighbor:
    print(item,"item")
    hard_case = item[-1]
    source_pred = hard_case["pred"]
    posi = 0
    neg = 0
    for i in range(len(item)-1):
        if item[i]["pred"] == 0:
            neg += 1
        else:
            posi += 1
    if neg > posi:
        hard_case["pred"] = 0
    else:
        hard_case["pred"] = 1
    if hard_case["pred"] != source_pred:
        hard_case["process"] = 1
        if hard_case["label"] == hard_case["pred"]:
            hard_case["is_correct"] = True
            key = hard_case["filename"] + str(hard_case["x_center"]) + str(hard_case["y_center"]) + str(hard_case["grid_idx"])
            changed_hard_case[key] = hard_case
        else:
            hard_case["is_correct"] = False

print("changed hardcase: ", len(changed_hard_case))
print("change hardcase class: {} s".format(time.time()-time2))

# save new res
time3 = time.time()
for line in source_res_json_lines:
    dic = json.loads(line)
    key = dic["filename"] + str(dic["x_center"]) + str(dic["y_center"]) + str(dic["grid_idx"])
    if key in changed_hard_case.keys():
        dic = changed_hard_case[key]
    new_res_json.write(json.dumps(dic) + "\n")
print("save new res: {} s".format(time.time()-time3))

print("all time: {} s".format(time.time()-start_time))
