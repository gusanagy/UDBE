import colorsys
import os
from typing import Dict, List
import PIL
import lpips as lpips
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import torch
import torch.optim as optim
#from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import albumentations as A
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet_Mask
from .Scheduler import GradualWarmupScheduler
from loss import Myloss
import numpy as np
from tensorboardX import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
from src.split_data import check_alpha_channel, load_image_paths
import torch.utils.data as data
import glob
import random
from albumentations.pytorch import ToTensorV2
import lpips
import time
import argparse
import wandb


class load_data_inf(data.Dataset):
    def __init__(self, input_data_low):
        self.input_data_low = input_data_low
        print("Total training examples:", len(self.input_data_low))
        self.transform=A.Compose(
            [
              # A.RandomCrop(height=400, width=400),
                A.Resize(400, 600),
                ToTensorV2(),
            ]
        )
        



    def __len__(self):
        return len(self.input_data_low)

    def __getitem__(self, idx):
        print('mask_path:',self.mask_path)
        mask=cv2.imread(self.mask_path)/255.
        mask=self.transform(image=mask)["image"]
        # mask=mask.permute(2, 0, 1)


        data_low = cv2.imread(self.input_data_low[idx])
        data_low = data_low[:,:,::-1].copy()
        random.seed(1)
        data_low = data_low/255.0


        data_low = self.transform(image=data_low)["image"]
        data_low2 = data_low
        data_low2 = data_low2*2-1
        data_low = torch.pow(data_low, 0.25)

        mean=torch.tensor([0.4350, 0.4445, 0.4086])
        var=torch.tensor([0.0193, 0.0134, 0.0199])
        data_low=(data_low-mean.view(3,1,1))/var.view(3,1,1)
        data_low=data_low/20


        data_max_r=data_low[0].max()
        data_max_g = data_low[1].max()
        data_max_b = data_low[2].max()
        color_max=torch.zeros((data_low.shape[0],data_low.shape[1],data_low.shape[2]))
        color_max[0,:,:]=data_max_r*torch.ones((data_low.shape[1],data_low.shape[2]))    #这里之前都写错了，应该从color_max[0:,:]改为color_max[0,:,:]#Isso foi escrito errado antes, deveria ser alterado de color_max[0:,:] para color_max[0,:,:]
        color_max[1,:, :] = data_max_g * torch.ones((data_low.shape[1], data_low.shape[2]))
        color_max[2,:, :] = data_max_b * torch.ones((data_low.shape[1], data_low.shape[2]))
        data_color=data_low/(color_max+ 1e-6)



        data_blur = data_low.permute(1, 2, 0).numpy() * 255.0
        data_blur = cv2.blur(data_blur, (5, 5))
        data_blur = data_blur * 1.0 / 255.0
        data_blur = torch.Tensor(data_blur).float().permute(2, 0, 1)

        mask = mask[0:1,:, :]
        return [data_low, data_color,data_blur,mask]

def getSnrMap(data_low,data_blur):
    data_low = data_low[:, 0:1, :, :] * 0.299 + data_low[:, 1:2, :, :] * 0.587 + data_low[:, 2:3, :, :] * 0.114
    data_blur = data_blur[:, 0:1, :, :] * 0.299 + data_blur[:, 1:2, :, :] * 0.587 + data_blur[:, 2:3, :, :] * 0.114
    noise = torch.abs(data_low - data_blur)

    mask = torch.div(data_blur, noise + 0.0001)

    batch_size = mask.shape[0]
    height = mask.shape[2]
    width = mask.shape[3]
    mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
    mask_max = mask_max.view(batch_size, 1, 1, 1)
    mask_max = mask_max.repeat(1, 1, height, width)
    mask = mask * 1.0 / (mask_max + 0.0001)

    mask = torch.clamp(mask, min=0, max=1.0)
    mask = mask.float()
    return mask


class load_data_test(data.Dataset):
    def __init__(self, input_data_low, input_data_high):
        self.input_data_low = input_data_low
        self.input_data_high = input_data_high
        print("Total test-training examples:", len(self.input_data_high))
        self.transform=A.Compose(
            [
                A.Resize (height=256, width=256),
                ToTensorV2(),
            ]
        )


    def __len__(self):
        return len(self.input_data_low)

    def __getitem__(self, idx):
        seed = torch.random.seed()

        data_low = cv2.imread(self.input_data_low[idx])

        #data_low = cv2.convertScaleAbs(data_low, alpha=1.0, beta=-75) #modificação para ajuste automatico de brilho para datalow

        ### Processamento das imagens // Avaliar aplicação // funcionalizar
        # Conversão de canais de cor e normalização
        data_low=data_low[:,:,::-1].copy()
        random.seed(1)
        data_low=data_low/255.0
        # Aplicação de correção gamma
        data_low=np.power(data_low,0.5)

        # Transformação e normalização
        data_low = self.transform(image=data_low)["image"]
        mean=torch.tensor([0.4350, 0.4445, 0.4086])
        var=torch.tensor([0.0193, 0.0134, 0.0199])
        #mean = torch.tensor([0.4174, 0.4897, 0.2255])
        #var = torch.tensor([0.0383, 0.0338, 0.0259])
        #mean=torch.tensor([0.2255, 0.4897, 0.4174])
        #var=torch.tensor([0.0259, 0.0338, 0.0383])
        data_low=(data_low-mean.view(3,1,1))/var.view(3,1,1)
        data_low=data_low/20

        # Calcular máximos dos canais de cor e normalização de cor
        data_max_r=data_low[0].max()
        data_max_g = data_low[1].max()
        data_max_b = data_low[2].max()
        color_max=torch.zeros((data_low.shape[0],data_low.shape[1],data_low.shape[2]))
        color_max[0,:,:]=data_max_r*torch.ones((data_low.shape[1],data_low.shape[2]))    
        color_max[1,:, :] = data_max_g * torch.ones((data_low.shape[1], data_low.shape[2]))
        color_max[2,:, :] = data_max_b * torch.ones((data_low.shape[1], data_low.shape[2]))
        data_color=data_low/(color_max+ 1e-6)
        #data_color = self.transform(data_color)
        #data_color=torch.from_numpy(data_color).float()
        #data_color=data_color.permute(2,0,1)

        # Processamento da imagem de alta luminosidade
        data_high = cv2.imread(self.input_data_high[idx])

        #data_high= cv2.convertScaleAbs(data_high, alpha=1.0, beta=50)

        data_high=data_high[:,:,::-1].copy()
        #data_high = Image.fromarray(data_high)
        random.seed(1)
        data_high = self.transform(image=data_high)["image"]/255.0
        data_high=data_high*2-1

        #normalization
        #data_high=data_high**0.25

        # Desfocagem e preparação do retorno
        data_blur = data_low.permute(1, 2, 0).numpy() * 255.0
        data_blur = cv2.blur(data_blur, (5, 5))
        data_blur = data_blur * 1.0 / 255.0
        data_blur = torch.Tensor(data_blur).float().permute(2, 0, 1)


        return [data_low, data_high,data_color,data_blur,self.input_data_low[idx]]


def Inference(config: Dict):
   ###load the data
    datapath_test = load_image_paths(config.dataset_path,config.dataset)[1]

    # load model and evaluate
    device = config.device_list[0]

    dataload_test = load_data_test(datapath_test,datapath_test)
    dataloader = DataLoader(dataload_test, batch_size=1, num_workers=4,
                            drop_last=True, pin_memory=True)

    model = UNet_Mask(T=config.T, ch=config.channel, ch_mult=config.channel_mult,
                 attn=config.attn,
                 num_res_blocks=config.num_res_blocks, dropout=0.)
    ckpt_path=config.pretrained_path
    ckpt = torch.load(ckpt_path,map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    print("model load weight done.")
    
    save_dir=config.output_path+'/result/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_txt_name =save_dir + 'res.txt'
    #f = open(save_txt_name, 'w+')
    #f.close()


    model.eval()
    sampler = GaussianDiffusionSampler(
        model, config.beta_1, config.beta_T, config.T).to(device)
    #loss_fn_vgg=lpips.LPIPS(net='vgg')

    with torch.no_grad():
        with tqdm( dataloader, dynamic_ncols=True) as tqdmDataLoader:
                image_num = 0
                for data_low, data_color,data_blur in tqdmDataLoader:
                    lowlight_image = data_low.to(device)

                    data_color = data_color.to(device)
                    data_blur=data_blur.to(device)
                    snr_map = getSnrMap(lowlight_image, data_blur)
                    
                    mask = mask.numpy()
                    # print('type:',type(mask))
                    print('data low shape:',data_low.shape)
                    print('mask shape:',mask.shape)
                    # mask=cv2.resize(mask,(data_low.shape[2],data_low.shape[3]))
                    mask=cv2.GaussianBlur(mask, (25, 25), 0)
                    mask=torch.from_numpy(mask).to(device)
                    # mask = torch.unsqueeze(mask, 0).to(device)
                    # mask = np.ones([400, 600,1]) * 0
                    # mask=torch.unsqueeze(mask,0).to(device)
                    # mask[:, :, 100:400, 100:400] = 1
                    
                    data_concate=torch.cat([data_color, snr_map,mask], dim=1)

                    for i in range(-8, 8): 
                        brightness_level = torch.ones([1]) * i
                        brightness_level = brightness_level.to(device)
                        
                        time_start = time.time()
                        sampledImgs = sampler(lowlight_image, data_concate,brightness_level,ddim=True,
                                            unconditional_guidance_scale=1,ddim_step=config.ddim_step)
                        time_end=time.time()
                        print('time cost:', time_end - time_start)

                        sampledImgs=(sampledImgs+1)/2
                        res_Imgs=np.clip(sampledImgs.detach().cpu().numpy()[0].transpose(1, 2, 0),0,1)[:,:,::-1] 

                        wandb.log({"Inference CLE_mask_DIF":
                                   {"Mask Image Inference": [wandb.Image(res_Imgs, caption="Image")],
                                   "Mask Image Inference *255": [wandb.Image(res_Imgs*255, caption="Image")]}})
                        save_path =save_dir+ config.data_name+'_level'+str(i)+'.png'
                        print(save_path)
                        cv2.imwrite(save_path, res_Imgs*255)
    


if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    modelConfig = {
        "DDP": False,
        "state": "train", # or eval
        "epoch": 60001,
        "batch_size":16 ,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-5,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:1",
        "device_list": [0],
        #"device_list": [3,2,1,0],
        
        "ddim":True,
        "unconditional_guidance_scale":1,
        "ddim_step":100
    }


    parser.add_argument('--dataset_path', type=str, default="./data/LOL/")
    parser.add_argument('--pretrained_path', type=str, default=None)  
    parser.add_argument('--output_path', type=str, default="./output_mask_inference/")
    parser.add_argument('--input_path', type=str, )
    parser.add_argument('--mask_path', type=str, )
    parser.add_argument('--data_name', type=str, help='output image name',default='result')
    
    config = parser.parse_args()
    
    for key, value in modelConfig.items():
        setattr(config, key, value)
    print(config)
    wandb.init(
        project="CLEDiffusion",
        config=vars(config),
        name="Inferencia mask Diffusao underwater",
        tags=["Inference"],
        group="diffusion_mask_inference",
        job_type="evaluation",
    )
    Test(config)
