# **Modified CLE Diffusion: Controllable Light Enhancement Diffusion Model for underwater color correction**
Authors: 

Institutions: 

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


## Uso do Dataset

Este dataset é organizado para facilitar o treinamento e a validação de modelos de aprendizado de máquina. As imagens estão divididas em conjuntos de treino (`train`) e validação (`val`) para cada categoria específica. Certifique-se de utilizar as imagens dos diretórios correspondentes conforme necessário para seus experimentos.


# Checkpoint

# Setup
```python
pip install -r requirements.txt
```

or 

```
conda env create -f CLEDiff_bkp.yaml
```

for conda envoiriments

# Usage
Our diffusion code structure is based on the original implementation of DDPM. Increasing the size of the U-Net may lead to better results.

About training iteration. The training with 5000 iterations has converged quite well. We recommend training for 10,000 iterations to achieve better performance, and you can select the best-performing training iterations.

We test code on one RTX 3090 GPU. The training time is about 1-2 days.
```python
python train.py   #train from scratch, you can change setting in modelConfig 
python train.py --pretrained_path ckpt/lol.pt  
python test.py --pretrained_path ckpt/lol.pt  
```

# Mask CLE Diffusion
Mask CLE Diffusion finetunes lol checkpoint. In our experiments, lol checkpoint is better than mit-adobe-5K checkpoint.

We show some inference cases in 'data/Mask_CLE_cases'. Welcome to use your cases to test the performance.

```python
python mask_generation.py   #generate masks for training
python train_mask.py --pretrained_path ckpt/lol.pt  #finetune Mask CLE Diffusion
python test_mask.py --pretrained_path ckpt/Mask_CLE.pt --input_path data/Mask_CLE_cases/opera.png --mask_path data/Mask_CLE_cases/opera_mask.png --data_name opera
```



# Acknowledgement
This work is mainly built on [DenoisingDiffusionProbabilityModel-ddpm](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-). Thanks a lot to authors for sharing!

# Citation

