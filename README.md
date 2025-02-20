# **BRDM: Brightness Restoration in Underwater Images from Diffusion Models**
Authors: Tatiana Taís Schein, Gustavo Pereira de Almeira, Stephanie Loi Brião, Rodrigo Andrade de Bem, Felipe Gomes de Oliveira and Paulo L. J. Drews-Jr.

Institutions: Universidade Federal do Rio Grande and Universidade Federal do Amazonas.

# Data

This dataset is organized to facilitate the training and validation of machine learning models. The images are divided into training (`train`) and validation (`val`) sets for each specific category. Make sure to use the images from the corresponding directories as needed for your experiments.

 ```
 UWSData
├── RUIE
│    ├── UCCS
│    |    ├── train
│    |    └── val
│    ├── UIQS
│    |    ├── train
│    |    └── val
│    └── UTTS
│         ├── train
│         └── val
├── SUIM
│    |── train
|    |    ├── images
|    |    └── masks
│    └── val
|         ├── images
|         └── masks
└── UIEB
      ├── train
      └── val
``` 

# Setup

First, create the environment with the necessary requirements:

conda create --name CLE pillow scikit-image matplotlib numpy tensorboardX tensorboard
conda activate CLE
conda install pytorch torchvision torchaudio cudatoolkit=12.1 -c pytorch
pip install lpips albumentations kornia gdown opencv-python wandb

# Usage
<!--Our diffusion code structure is based on the original implementation of DDPM. Increasing the size of the U-Net may lead to better results. About training iteration. The training with 5000 iterations has converged quite well. We recommend training for 10,000 iterations to achieve better performance, and you can select the best-performing training iterations.We test code on one RTX 3090 GPU. The training time is about 1-2 days.*/ -->
Our code was trained on a computer with two NVIDIA TITAN X GPUs, totaling 24GB of GPU memory.

```
python #train from scratch, you can change setting in modelConfig 
python main.py --dataset_path "" --dataset "UIEB" -- state "train" 
python main.py --dataset_path "" --dataset "SUIM" -- state "train" 

python main.py --pretrained_path  --dataset "SUIM" -- state "eval" --pretrained_path "1000.pt"
python main.py --pretrained_path  --dataset "UIEB" -- state "eval" --pretrained_path "1000.pt"

python main.py --pretrained_path  --dataset "SUIM" -- state "inference" --inference_image " "  --pretrained_path "1000.pt"
python main.py --pretrained_path  --dataset "UIEB" -- state "inference" --inference_image " " --pretrained_path "1000.pt"


```
# Testing and Visualization
<!--
Run the script ('avaliacao.ipynb') . This script will automatically download the dataset and weights, and it includes a cell to run inference for each dataset. Similarly, the visualization of the images from the respective datasets can be done using this script.
*/ -->

# ACKNOWLEDGMENT
This study was funded, in part, by the São Paulo Research Foundation (FAPESP), Brazil, under Process Number 2024/10523-5. The authors would also like to thank the PRH-ANP and CNPQ organizations for their research support and financial assistance.
