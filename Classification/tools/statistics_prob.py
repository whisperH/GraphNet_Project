import os
import json
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX

# root = "/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/res34_69tif_1cls_v2.1.0/test_result_ckpt20_thresh05.json"
root = "/home/huangxiaoshuang/medicine/HCC_Prognostic/logs/classification/ckpt/res34_69tif_1cls_v2.1.0/process_res/infer_v2_test_result_ckpt20_thresh05.json"
file0 = open(root,'r',encoding="utf-8")
lines0 = file0.readlines()
file0.close()

count = 0

path_lst = root.split("/")
save_path = os.path.join("/"+"/".join(path_lst[:-1]), "draw_prob")
os.makedirs(save_path, exist_ok=True)

prob_list = []
tp = 0
fp = 0
for line in lines0:
    grid = json.loads(line)
    flag = False # True==tp, False==fp
    if grid["pred"]:
        if grid["label"]:
            flag = True
            tp += 1
        else:
            fp += 1
        prob_list.append([grid["pred_logit"], flag])
prob_list.sort(key=lambda k: k[0])
print("len cancer prob", len(prob_list))
print("tp: {}, fp: {}".format(tp, fp))

tp_list = []
fp_list = []
for idx,item in enumerate(prob_list):
    if item[1]:
        tp_list.append([idx, item[0]])
    else:
        fp_list.append([idx, item[0]])
print("tp_list: {}, fp_list: {}".format(len(tp_list), len(fp_list)))
tp_list = np.array(tp_list)
fp_list = np.array(fp_list)
# plt.figure(figsize=(20,10))
# plt.figure()
############################ draw prob ############################

plt.scatter(tp_list[:,0], tp_list[:,1], c="g", s=5)
plt.scatter(fp_list[:,0], fp_list[:,1], c="r", s=1)
plt.grid(True)

f = plt.gcf()  #获取当前图像
f.savefig(save_path+'/{}.png'.format("pred_cancer_prob_sort_v3"))
f.clear()  #释放内存

############################ draw prob individually ################################
plt.figure(figsize=(30,10))
plt.subplot(121)
plt.scatter(tp_list[:,0], tp_list[:,1], c="g", s=1)
plt.grid(True)
plt.subplot(122)
plt.scatter(fp_list[:,0], fp_list[:,1], c="r", s=1)
plt.grid(True)
f = plt.gcf()  #获取当前图像
f.savefig(save_path+'/{}.png'.format("pred_cancer_prob_sort_v4"))
f.clear()  #释放内存

############################ draw prob histogram individually ############################
bins_lst =[i/10 for i in range(0,11,1)]
bias = 0.04
# plt.figure(figsize=(20,10))
plt.subplot(121)
nums,bins,patches = plt.hist(tp_list[:,1], bins=bins_lst, edgecolor='k')
# plt.xticks(bins,bins)
# plt.grid(True)
for num,bin in zip(nums,bins):
    plt.annotate(int(num),xy=(bin,num),xytext=(bin+bias,num+3))

plt.subplot(122)
nums,bins,patches = plt.hist(fp_list[:,1], bins=bins_lst,edgecolor='k')
print(bins)
for num,bin in zip(nums,bins):
    plt.annotate(int(num),xy=(bin,num),xytext=(bin+bias,num+3))
f = plt.gcf()  #获取当前图像
f.savefig(save_path+'/{}.png'.format("pred_cancer_prob_sort_v5"))
f.clear()  #释放内存

############################ draw prob histogram ############################
bins_lst =[i/20 for i in range(0,21,1)]
bias = 0.015
plt.figure(figsize=(20,10))
nums,bins,patches = plt.hist(tp_list[:,1], bins=bins_lst, edgecolor='k')
for num,bin in zip(nums,bins):
    plt.annotate(int(num),xy=(bin,num),xytext=(bin+bias,num+150))
nums,bins,patches = plt.hist(fp_list[:,1], bins=bins_lst, edgecolor='k')
for num,bin in zip(nums,bins):
    plt.annotate(int(num),xy=(bin,num),xytext=(bin+bias,num//2), color='r')
plt.grid(True)
f = plt.gcf()  #获取当前图像
f.savefig(save_path+'/{}.png'.format("pred_cancer_prob_sort_v6"))
f.clear()  #释放内存

