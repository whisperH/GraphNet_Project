# Overview
graph-net project aims to be a tools to compute a tumor risk score (G-TRS) 
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
graph-net mainly depends on the Python scientific stack which is listed in the file :[enviroment.yml](enviroment.yml).

### Installation Guide:
we use conda for building enviroment, if you are unfamiliar with conda, 
please refer this link https://www.anaconda.com/docs/getting-started/miniconda/install#macos-linux-installation
#### Install from Github
```bash
git clone https://github.com/whisperH/GraphNet_Project.git
cd GraphNet_Project
conda env create -f environment.yml
```

### Execution Guide:
Data, Weights and Log Link: https://pan.baidu.com/s/1V0l5kYQ8C6_9UVfBRWAN8Q?pwd=1234 code: 1234
- If you wish to identify cancer regions in WSI slides, please navigate to the classification folder and follow the instructions in the readme.md file within it.
- If you have already obtained the slice images of cancer regions, please navigate to the Prognostic folder and follow the instructions in the readme.md file within it.

# Our implementation refers the following publicly available codes.
- Shi JY，Wang X，Ding GY，et al.Exploring prognostic indicators in the pathological images of hepatocellular carcinoma based on deep learning［J］.Gut，2021，70（5）：951-961.
- Pytorch Geometric--Fey M, Lenssen J E. Fast graph representation learning with PyTorch Geometric[J]. arXiv preprint arXiv:1903.02428, 2019.
- Hou, W. et al. (2023). Multi-scope Analysis Driven Hierarchical Graph Transformer for Whole Slide Image Based Cancer Survival Prediction. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14225. Springer, Cham. https://doi.org/10.1007/978-3-031-43987-2_72
- Lu, M.Y., Chen, B., Zhang, A., Williamson, D.F., Chen, R.J., Ding, T., Le, L.P., Chuang, Y.S. and Mahmood, F., 2023. Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 19764-19775).

# License
This project is covered under the Apache 2.0 License.