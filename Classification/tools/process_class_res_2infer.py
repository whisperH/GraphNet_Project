import os
import json
import numpy as np
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
import time
from tqdm import tqdm

start_time = time.time()


# build new res json path
save_root = os.path.join("/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/1cls_v4.0.4/", "process_res")
save_json_name = "infer_v2_test_result_ckpt76_thresh06.json"
os.makedirs(save_root, exist_ok=True)

# search radius ratio
grid_size = 256
radius = 2
kernel = pow(int(radius)*2+1, 2) - 1
distance = grid_size*radius
# read infer res json
source_res_json_path = "/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/1cls_v4.0.4/process_res/infer_v1_test_result_ckpt76_thresh06.json"
source_res_json = open(source_res_json_path, 'r', encoding="utf-8")
source_res_json_lines = source_res_json.readlines()
source_res_json.close()



new_res_json = open(os.path.join(save_root, save_json_name), 'w', encoding="utf-8")

time0 = time.time()
# get all hard grid
hard_grid_list = []
all_grid_dic = {}
for line in source_res_json_lines:
    dic = json.loads(line)
    tif_name = dic["filename"].split("_")[0]
    left_top_x = dic["x_center"] - 384
    left_top_y = dic["y_center"] - 384
    grid_center_x = dic["grid_idx"]//3*256 + left_top_x + grid_size // 2
    grid_center_y = dic["grid_idx"]%3*256 + left_top_y + grid_size // 2
    dic["grid_center_x"] = grid_center_x
    dic["grid_center_y"] = grid_center_y
    
    if not dic['is_correct']:
        hard_grid_list.append(dic)
    if tif_name not in all_grid_dic.keys():
        all_grid_dic[tif_name]=[dic]
    else:
        all_grid_dic[tif_name].append(dic)

print("all_grid_dic: ", len(all_grid_dic))
print("get all grid: {} s".format(time.time()-time0))

# sort all grid per tif by filename
for k, v in all_grid_dic.items():
    v = sorted(v, key=lambda item: item["grid_center_x"])
    all_grid_dic[k] = v
# for idx, i in enumerate(all_grid_dic["D17-02013-01"]):
#     print(i)
    # if idx > 200:
    #     break

# get hard case and its neighbor
time1 = time.time()
grid_and_neighbor = []
bar = tqdm(all_grid_dic.items())
for filename, grid_list in bar:
    print(filename)
    lenght = len(grid_list)
    print(lenght)
    idx = 0
    for grid in tqdm(grid_list):
    # for grid in grid_list:
        grid_center_x = grid["grid_center_x"]
        grid_center_y = grid["grid_center_y"]
        neighbor = []
        # print(len(all_grid_dic[grid["filename"]]), "----")
        start = max(0, idx - 10000)
        end = min(lenght, idx + 10000)
        for i in range(0, lenght):
            if (grid_center_x-distance)<grid_list[i]["grid_center_x"]<(grid_center_x+distance) and (grid_center_y-distance)<grid_list[i]["grid_center_y"]<(grid_center_y+distance):
                neighbor.append(grid_list[i])
                if len(neighbor) >= kernel:
                    break
        # whether grid has 9 neighbors
        if len(neighbor) > 5:
            neighbor.append(grid)
            grid_and_neighbor.append(neighbor)
    idx += 1
# bar = tqdm(hard_grid_list)
# for grid in bar:
#     idx = 0
#     grid_center_x = grid["grid_center_x"]
#     grid_center_y = grid["grid_center_y"]
#     neighbor = []
#     for neighbor_ in all_grid_dic[grid["filename"].split("_")[0]]:
#         if (grid_center_x-distance)<neighbor_["grid_center_x"]<(grid_center_x+distance) and (grid_center_y-distance)<neighbor_["grid_center_y"]<(grid_center_y+distance):
#             neighbor.append(neighbor_)
#             if len(neighbor) >= kernel:
#                 break
#     # whether grid has 9 neighbors
#     if len(neighbor) > 5:
#         neighbor.append(grid)
#         grid_and_neighbor.append(neighbor)
#     idx += 1

# print("grid_and_neighbor: ", len(grid_and_neighbor))
# print("get grid case and its neighbor: {} s".format(time.time()-time1))

# change hardcase class
time2 = time.time()
changed_grid = {}
num_true = 0
num_false = 0
num_no_vari = 0
for item in grid_and_neighbor:
    # print(item,"item")
    grid = item[-1]
    source_pred = grid["pred"]
    posi = 0
    neg = 0
    for i in range(len(item)-1):
        if item[i]["pred"] == 0:
            neg += 1
        else:
            posi += 1
    if neg > posi:
        grid["pred"] = 0
    elif neg == posi:
        grid["pred"] = source_pred
    else:
        # print(neg, posi)
        grid["pred"] = 1
    if grid["pred"] != source_pred:
        grid["process"] = 1
        if grid["label"] == grid["pred"]:
            grid["is_correct"] = True
            key = grid["filename"] + str(grid["x_center"]) + str(grid["y_center"]) + str(grid["grid_idx"])
            changed_grid[key] = grid
            num_true += 1
        else:
            grid["is_correct"] = False
            num_false += 1
    else:
        num_no_vari += 1

print("num_true: ", num_true, "num_false:", num_false, "num_no_vari:", num_no_vari)
print("change hardcase class use: {} s".format(time.time()-time2))

# save new res
time3 = time.time()
for line in source_res_json_lines:
    dic = json.loads(line)
    key = dic["filename"] + str(dic["x_center"]) + str(dic["y_center"]) + str(dic["grid_idx"])
    if key in changed_grid.keys():
        dic = changed_grid[key]
    new_res_json.write(json.dumps(dic) + "\n")
print("save new res: {} s".format(time.time()-time3))

print("all time: {} s".format(time.time()-start_time))
