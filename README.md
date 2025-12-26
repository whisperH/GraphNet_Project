# Overview
Graph-net project aims to be a tools to compute a tumor risk score (G-TRS) 
designed to refine current transplant eligibility standards. 
By combining graph networks with instance and batch normalization layers, 
GraphNet effectively captures critical features of tumor heterogeneity 
and the surrounding microenvironment, ensuring consistent comparability 
of G-TRS across different centers. The package can be installed on GNU/Linux from Python Package Index
(PyPI) and GitHub.

# System Requirements
## Hardware requirements
graph-net requires only a standard computer with enough RAM to support the in-memory operations.
Note that if you have a GPU graphics card, you will achieve faster inference speeds. 
Otherwise, the CPU will be used for inference, which will be slightly slower in speed but can still attain the same results.

## Software requirements
### OS Requirements
This package is supported for Linux. The package has been tested on the following systems:
- Linux: Ubuntu 24.04.1 LTS
### Python Dependencies
Graph-net mainly depends on the Python scientific stack which is listed in the file :[enviroment.yml](enviroment.yml).

### Installation Guide:
we use conda for building enviroment, if you are unfamiliar with conda, 
please refer this link https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation
#### Install from Github
```bash
git clone https://github.com/whisperH/GraphNet_Project.git
cd GraphNet_Project
conda env create -f environment.yml
```

## Execution Guide:
The Structure of file is:

    ```bash
    # 1.1 Classification Data Preprocess
    -- /raw_data_folder/TraingModel
        -- train
            -- Tumor
                -- xxx.png
                -- ...
            -- NonTumor
                -- 123.png
                -- ...
        -- val
            -- Tumor
                -- xxx1.png
                -- ...
            -- NonTumor
                -- 1231.png
                -- ...
        -- test
            -- Tumor
                -- xx2x.png
                -- ...
            -- NonTumor
                -- 1232.png
                -- ...
    # 1.3 Classification Model Infering

    -- /raw_data_folder/CenterName
        -- WSI-A.svs
        -- WSI-B.tif
        -- ...

    # 2. Patch Encoding with Conch Model
    -- /folder_to_save_process_data/patchImage
        -- CenterName                                                 
            -- WSI-A
                -- WSI_a_x1_y1_conf1.png
                -- WSI_a_x2_y2_conf2.png
                -- ...
            -- WSI-B
            -- ...
    # Generated when step2 has been finished.
    -- /folder_to_save_process_data/patchEmbedding
        -- CenterName                                                
            -- WSI-A
                -- WSI_a_x1_y1_conf1.npz
                -- WSI_a_x2_y2_conf2.npz
                -- ...
            -- WSI-B
            -- ...
    # 3. Construct graph-structured data
    -- /folder_to_save_process_data/patchImage/CenterName
    -- /folder_to_save_process_data/patchEmbedding/CenterName
    -- /Prognostic_dir/rfs_CenterName.json
    
    # 4. Performing prognostic score analysis based on the patch results
    # Generated when step3 has been finished.
    -- /folder_to_save_process_data/GraphData
        -- rfs_CenterName.pt               # the graph data with pyg format
        -- rfs_CenterName_GP_maps.json     # storing the mapping information of patient and WSI ID

    # The results will be saved in 
    -- /out_result_dir/
        -- CIndex_[time-str].txt            # C-Index for testing model
        -- rfs_[CenterName].csv             # score for each patient
    ```

### Using the trained GraphNet model:
The computation of WSI required for liver transplantation into G-TRS involves the following steps:
1. **The WSI is divided into patches, and a trained classification model is used to score the tumor regions**.

    1.1 **Classification Data Preprocess**: A generic data loader where the images are arranged in this way by default.

    For the specific tools and methods to convert .tif, .svs, and .mrxs whole slide images (WSI) into 768x768 patches, you can refer to the following resources and approaches:[OpenSlide](https://openslide.org/api/python/)以及[HCC_Prognostic
](https://github.com/wangxiaodong1021/HCC_Prognostic).

    > Note: In this study, the classification model incorporates information from the surrounding patches when determining the category of the central patch. To facilitate this, the model's input is a larger region of 768x768 pixels. However, within the model's architecture, this 768x768 input is subsequently divided and processed as standard 224x224 images. For specific implementation details, please refer to the corresponding section of the code.

    1.2 **Classification Model training and testing**
    ```bash
    cd ./dir-to-GraphNet_Project/Classification
    python train_ImageFolder.py -c ./path_to_config_folder/config.yaml

    # e.g.
    # python train_ImageFolder.py -c ./work_dirs/LC25000Colon/config.yaml
    # python test_ImageFolder.py -c ./work_dirs/LC25000Colon/config.yaml
    ```

    1.3 **Classification Model Infering**

    The command requires several arguments:
    - ```-c``` represents the configuration file to be used;
    - ```-i``` represents the folder containing the WSI data to be inferred;
    - ```-s``` represents the folder where the patches are stored;
    - ```-n``` represents the number of patches stored.

    ```bash

    python infer_on_dir.py -c ./path_to_config_folder/config.yaml -i raw_data_folder/CenterName -s folder_to_save_process_data -n number_of_patches_to_save
    # e.g.
    python infer_on_dir.py -c ./path_to_config_folder/config.yaml -i ./raw_data_folder/CenterName -s /folder_to_save_process_data -n 20
    ```
    
    The command will create a folder named ```CenterName``` within the ```folder_to_save_process_data``` directory. 
    This ```CenterName``` is automatically assigned by the script "infer_on_dir.py" and is taken directly from the name of the last-level subdirectory in your ```folder_to_CenterName``` path. 
    Inside this ```CenterName``` folder, the ```number_of_patches_to_save``` highest-scoring patches for each Whole Slide Image (WSI) from ```./data/Test``` will be stored.


2. **Patch Encoding with Conch Model**: The Conch large language model is used to encode the selected patch images.

    The command requires several arguments:
    - ```--parent_dir```: the folder_to_save_process_data.
    - ```--dataset_name```: the ```CenterName``` folder created in the previous step (e.g., "Test").
    - ```--patch_embedding_dir```: the folder designated to store the final encoded results.

    ```bash
    cd ./dir-to-GraphNet_Project/Prognostic/data_preprocess

    python step3_multi_infer_WSI_HCC_patch_feat1.py 
    --parent_dir folder_to_save_process_data 
    --dataset_name CenterName 

    # e.g. 
    python step3_multi_infer_WSI_HCC_patch_feat1.py --parent_dir folder_to_save_process_data --dataset_name Test
    ```

    
3. **Construct graph-structured data**

    The command requires several arguments:
    - ```--parent_dir```: the folder_to_save_process_data.
    - ```--dataset_name```: the ```CenterName``` folder created in the previous step (e.g., "Test").
    - ```--graph_file_saved_path```: the folder designated to store the graph data.
    - ```-- link```: the connection method of graph data.
    - ```--json_dir```: The path to the JSON file that stores the mapping between patients and WSI slides. If one patient corresponds to multiple WSI slides, this parameter must be used to read the JSON file. Otherwise, it defaults to None.

    > Note: the json file format is listed as:

    3.1 For training the WSI data by prognostic model, the mapping json file named with ```rfs_[CenterName].json```
    ```json
    rfs_Test.json

    [
        {
            "RFS_daytime": 1950,          # This parameter is used in training.
            "RFS_time": 65.0,             # This parameter is used in training.
            "events": 0,                  # This parameter is used in training.
            "PID": "CY1",                 # Unique WSI ID
            "path": "CY/CY1",             # WSI ID of patient: CenterName/PID
            "name": "CY1"                 # Name of patient: a single name can correspond to one or multiple WSI IDs
        },
        {
            "RFS_daytime": 1410, 
            "RFS_time": 47.0, 
            "events": 0, 
            "PID": "CY20", 
            "path": "CY/CY20", 
            "name": "CY20"
        }...
    ]
    ```

    3.2 For infering the WSI data by prognostic model, the mapping json file named with ```rfs_[CenterName].json```
    ```json
    rfs_Test.json

    [
        {
            "RFS_daytime": -1,          # This parameter is unused in inference.
            "RFS_time": -1,             # This parameter is unused in inference.
            "events": -1,               # This parameter is unused in inference.
            "PID": "CY20",              # Unique WSI ID
            "path": "CY/CY20",          # WSI ID of patient: CenterName/PID
            "name": "CY20"              # Name of patient: a single name can correspond to one or multiple WSI IDs
        },
        {
            "RFS_daytime": -1,
            "RFS_time": -1,
            "events": -1,
            "PID": "CY1",
            "path": "CY/CY1",
            "name": "CY1"
        }...
    ]
    ```

    ```bash
    cd ./dir-to-GraphNet_Project/Prognostic/data_preprocess

    python step4_generate_instance_graph.py 
    --parent_dir folder_to_save_process_data 
    --json_dir Prognostic_dir 
    --dataset_name CenterName 
    --graph_file_saved_path GraphData 
    --link near8

    # e.g. 
    python step4_generate_instance_graph.py --parent_dir /home/server/code/GraphNet_Project/data --json_dir Prognostic_dir --dataset_name Test --graph_file_saved_path GraphData --link near8
    ```




4. **Performing prognostic score analysis based on the patch results**
    4.1 **Training your own prognostic model
    ```bash
    cd ./dir-to-GraphNet_Project/Prognostic

    python train/train_batch_1000.py --parent_dir folder_to_save_process_data  --graph_file_saved_path GraphData --log_result_dir ./logs/GraphNet100 --use_gnn_norm --exp_times 100
    ```

    4.2 **Inferring the trained prognostic model**
    The command requires several arguments:
    - ```--use_gnn_norm```: Using MFA Module of GraphNet.
    - ```--parent_dir```: the folder_to_save_process_data.    
    - ```--data_inferlists```: the ```CenterName``` folder created in the previous step (e.g., "Test").
    - ```--graph_file_saved_path```: the folder designated to store the graph data.
    - ```--finished_model```: the final weights of GraphNet for calculate the G-TRS
    - ```--out_result_dir```: The path to save G-TRS resutls


    ```bash
    cd ./dir-to-GraphNet_Project/Prognostic/train

    python InferGTRS.py --use_gnn_norm --parent_dir folder_to_save_process_data --graph_file_saved_path GraphData --data_inferlists CenterName --finished_model /path_to_GraphNetModel/GraphNet_Model.pth --out_result_dir /out_result_dir/
    ```
### Using R code for statistical analysis:
Regarding the omic data analysis in the paper, we have provided corresponding R code in the folder of "R_Statistical_Analysis", including statistical analysis, KM, Cox and Clinical Analysis, as well as Proteomic analysis. The three files contain detailed step-by-step instructions, and the corresponding data can be found in the data attachments provided in the published paper.

# The dataset and weights information
## The Classification dataset
- Lung and Colon Cancer Histopathological Images.([Link](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images))
- Histological images for tumor detection in gastrointestinal cancer.([Link](https://zenodo.org/records/2530789))

## The Prognoistic dataset
- The Graph Data of 20 patches and the weights.([GraphNet Information](https://drive.google.com/drive/folders/1Yr9xDQbbG5oi9EdMXzcy3cZGt54ALPDw?usp=sharing))

# Our implementation refers the following publicly available codes.
- Shi JY，Wang X，Ding GY，et al.Exploring prognostic indicators in the pathological images of hepatocellular carcinoma based on deep learning［J］.Gut，2021，70（5）：951-961.
- Pytorch Geometric--Fey M, Lenssen J E. Fast graph representation learning with PyTorch Geometric[J]. arXiv preprint arXiv:1903.02428, 2019.
- Hou, W. et al. (2023). Multi-scope Analysis Driven Hierarchical Graph Transformer for Whole Slide Image Based Cancer Survival Prediction. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14225. Springer, Cham. https://doi.org/10.1007/978-3-031-43987-2_72
- Lu, M.Y., Chen, B., Zhang, A., Williamson, D.F., Chen, R.J., Ding, T., Le, L.P., Chuang, Y.S. and Mahmood, F., 2023. Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 19764-19775).
- Monod M, Krusche P, Cao Q, et al. Torchsurv: A lightweight package for deep survival analysis[J]. arXiv preprint arXiv:2404.10761, 2024.
# License
This project is covered under the Apache 2.0 License.
