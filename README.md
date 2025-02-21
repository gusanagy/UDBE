# **BRDM: Brightness Restoration in Underwater Images from Diffusion Models**
<!--
Adicionar o link do lattes em todos 
-->
Tatiana Tais Schein, tatischein@furg.br | Gustavo Pereira de Almeira, gustavo.pereira.furg@furg.br | Stephanie Loi Briao, stephanie.loi@furg.br | Felipe Gomes de Oliveira, felipeoliveira@ufam.edu.br | Rodrigo Andrade de Bem, rodrigo.bem@gmail.com | Paulo L. J. Drews-Jr, paulodrews@furg.br



Activities in underwater environments are paramount in several scenarios, which drives the continuous development of underwater image enhancement techniques. A major challenge in this domain is the depth at which images are captured, with increasing depth resulting in a darker environment. Most existing methods for underwater image enhancement focus on noise removal and color adjustment, with few works dedicated to brightness  enhancement. This work introduces a novel unsupervised learning approach to underwater image enhancement using a diffusion model. Our method, called UDBE, is based on conditional diffusion to maintain the brightness  etails of the unpaired input images. The input image is combined with a color map and a SignalNoise Relation map (SNR) to ensure stable training and prevent  color distortion in the output images. The results demonstrate that our approach achieves an impressive accuracy rate in the datasets UIEB, SUIM and RUIE, well-established underwater image benchmarks. Additionally, the experiments validate the robustness of our approach, regarding  he image quality metrics PSNR, SSIM, UIQM, and UISM, indicating the good performance of the brightness enhancement process.

## Data

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


## Dataset

Link to the dataset used UWDdata: [UWDdata](https://drive.google.com/file/d/1SCwOosZam8bzoZdVSwW60l-bD7c65pv0/view?usp=sharing)

Create a folder named `data` and extract the dataset into it. Move the `UWData` folder to the `data` directory. Test the code with the dataset by using the `--dataset` flag to select the desired dataset, or process all datasets by leaving this parameter unset. Unzip and load the checkpoints using the script's parameters.

* [UWData Download](https://drive.google.com/file/d/1SCwOosZam8bzoZdVSwW60l-bD7c65pv0/view?usp=sharing)


## Checkpoint

Our checkpoint file is stored in huggingface. You need to download the file, unzip the weights for our three datasets and move them to the output/ckpt paste for easy use of the pretrained weights. The notebook [avaliacao.ipynb](avaliacao.ipynb) can be used for test our data and download the weights properly to easy test. 

* [Download Checkpoints](https://huggingface.co/Gusanagy/UDBE-Unsupervised-Diffusion-based-Brightness-Enhancement-in-Underwater-Images/tree/main)


## Setup
```python
pip install -r requirements.txt
```

or 

```conda
conda env create -f CLEDiff_bkp.yaml --name CLEDiff
```

for conda envoiriments

## Usage
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

# ACKNOWLEDGMENT
This study was funded, in part, by the São Paulo Research Foundation (FAPESP), Brazil, under Process Number 2024/10523-5. The authors would also like to thank the PRH-ANP and CNPQ organizations for their research support and financial assistance.
