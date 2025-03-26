## 1. cmd for training, testing and inferring
```bash
python train.py -c ./work_dirs/1102_res34_69tif_1cls_v2.0.0/config.yaml
python test.py -c ./work_dirs/1102_res34_69tif_1cls_v2.0.0/config.yaml
python infer_on_dir.py -c ./work_dirs/1102_res34_69tif_1cls_v2.0.0/config.yaml -i folder_to_CenterName -s folder_to_save_patches -n number_of_patches_to_save
```
download the weights_file from https://pan.baidu.com/s/1V0l5kYQ8C6_9UVfBRWAN8Q?pwd=1234 code: 1234
put folder "logs" of "Classification" into GraphNet_Project/Classification

## 2. Our implementation refers the following publicly available codes.
- Shi JY，Wang X，Ding GY，et al.Exploring prognostic indicators in the pathological images of hepatocellular carcinoma based on deep learning［J］.Gut，2021，70（5）：951-961.