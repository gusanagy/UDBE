# **BRDM: Brightness Restoration in Underwater Images from Diffusion Models**
Authors: Tatiana Taís Schein, Gustavo Pereira de Almeira, Stephanie Loi Brião, Rodrigo Andrade de Bem, Felipe Gomes de Oliveira and Paulo L. J. Drews-Jr.

Institutions: Universidade Federal do Rio Grande and Universidade Federal do Amazonas.

# Data

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


## Using the Dataset

This dataset is organized to facilitate the training and validation of machine learning models. The images are divided into training (`train`) and validation (`val`) sets for each specific category. Make sure to use the images from the corresponding directories as needed for your experiments.

Link to the dataset used UWDdata: https://drive.google.com/file/d/1SCwOosZam8bzoZdVSwW60l-bD7c65pv0/view?usp=sharing

Create a folder named `data` and extract the dataset into it. Move the `UWData` folder to the `data` directory. Test the code with the dataset by using the `--dataset` flag to select the desired dataset, or process all datasets by leaving this parameter unset. Unzip and load the checkpoints using the script's parameters.


* [Download Checkpoints](https://huggingface.co/Gusanagy/UDBE-Unsupervised-Diffusion-based-Brightness-Enhancement-in-Underwater-Images/tree/main)


# Checkpoint

# Setup
```python
pip install -r requirements.txt
```

or 

```conda
conda env create -f CLEDiff_bkp.yaml --name CLEDiff
```

for conda envoiriments

# Usage
<!--Our diffusion code structure is based on the original implementation of DDPM. Increasing the size of the U-Net may lead to better results. About training iteration. The training with 5000 iterations has converged quite well. We recommend training for 10,000 iterations to achieve better performance, and you can select the best-performing training iterations.We test code on one RTX 3090 GPU. The training time is about 1-2 days.*/ -->
 Nosso codigo foi treinado em um computador com duas placas NVIDIA TITAN X com 24gb de gpu no total.  

```
python #train from scratch, you can change setting in modelConfig 
python main.py --dataset_path "" --dataset "UIEB" -- state "train" 
python main.py --dataset_path "" --dataset "SUIM" -- state "train" 

python main.py --pretrained_path  --dataset "SUIM" -- state "eval" --pretrained_path "1000.pt"
python main.py --pretrained_path  --dataset "UIEB" -- state "eval" --pretrained_path "1000.pt"

python main.py --pretrained_path  --dataset "SUIM" -- state "inference" --inference_image " "  --pretrained_path "1000.pt"
python main.py --pretrained_path  --dataset "UIEB" -- state "inference" --inference_image " " --pretrained_path "1000.pt"


```

Os testes podem ser feitos no notebook [avaliacao.ipynb](avaliacao.ipynb). Assim como a visualização das imagens dos respectivos datasets. Ao rodar este notebook existem opções para baixar e gerar automaticamente as pastas, datasets e checkpoints necessários para rodar o modelo.
<!--
# Mask CLE Diffusion
Mask CLE Diffusion finetunes lol checkpoint. In our experiments, lol checkpoint is better than mit-adobe-5K checkpoint.

We show some inference cases in 'data/Mask_CLE_cases'. Welcome to use your cases to test the performance.
 /*We show some inference cases in 'data/Mask_CLE_cases'. Welcome to use your cases to test the performance.

```python
python mask_generation.py   #generate masks for training
python train_mask.py --pretrained_path ckpt/lol.pt  #finetune Mask CLE Diffusion
python test_mask.py --pretrained_path ckpt/Mask_CLE.pt --input_path data/Mask_CLE_cases/opera.png --mask_path data/Mask_CLE_cases/opera_mask.png --data_name opera
```
*/ -->

