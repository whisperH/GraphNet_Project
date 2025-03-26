## 1. data prepare

if you want to use the raw patch data, following the pre-process data, otherwise using the dataset/GraphDataNone and jumping to 2. train model

Data, Weights and Log Link: https://pan.baidu.com/s/1V0l5kYQ8C6_9UVfBRWAN8Q?pwd=1234 code: 1234

### Pre-Process data

1. download RFS annotation file (RFS_Data_UpLoad) from url and patch images (CY,huashan,huashan2,hz,JiangData32,YouAn)

2. patch embedding:

   * create the env of (CONCH)[https://github.com/mahmoodlab/CONCH]  and its weights file

   * modify the variation of "parent_dir" and "json_dir"

   * run data_preprocess/step3_infer_patch_feat.py and you will get the **.npz corresponding to patch image

3. run data_preprocess/step4_generate_instance_graph.py and you will get the rfs_XX.pt and rfs_XX_GP_maps.json under the dataset/GraphDataNone.

The Structure of file is:
```
--dataset
   -- patch_images;
   -- CY
      -- CY1
         -- CY1_XXXX_XXXX_XXXX.npz
         -- CY1_XXXX_XXXX_XXXX.png
         -- .....
      -- huashan;
      -- huashan2;
      -- hz;
      -- JiangData32;
      -- YouAn
   -- RFS_Data_UpLoad
   -- GraphDataNone
      -- rfs_CY.pt
      -- rfs_CY_GP_maps.json
      -- .....
--logs
```

## 2. train model

#### 2-1. Search the best parameters of model

```bash
python train/train_ema_search_para.py
```

#### 2-2. Using the best parameters of model as the hyper-paremeters

```bash
python train/train_batch_ema.py
```
The final weights and training log will be saved in the logs/log_result/EMA_GIN_ALL70_TimeFit

#### 2-3. The trained weights and logs are listed in the Logs.zip

## 3. Infer model
for infer data:
```bash
python train/train_batch_ema.py --infer_flag --PostProcess AverageGFeat
```
The results will be saved in the logs/log_result/EMA_GIN_ALL70_TimeFit

## 4. Our implementation refers the following publicly available codes.
- Pytorch Geometric--Fey M, Lenssen J E. Fast graph representation learning with PyTorch Geometric[J]. arXiv preprint arXiv:1903.02428, 2019.
- Hou, W. et al. (2023). Multi-scope Analysis Driven Hierarchical Graph Transformer for Whole Slide Image Based Cancer Survival Prediction. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14225. Springer, Cham. https://doi.org/10.1007/978-3-031-43987-2_72
