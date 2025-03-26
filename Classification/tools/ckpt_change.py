from turtle import st
import torch
state_dict = torch.load("/home/guozihao/gzh_medical_image_classify/model/MobileNetV2_0/best.ckpt")
torch.save(state_dict,"/home/guozihao/gzh_medical_image_classify/model/MobileNetV2_0/best_1.ckpt",_use_new_zipfile_serialization=False)