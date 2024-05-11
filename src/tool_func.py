import torch
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM    
# getSnrMap
# rgb2gray
# get_color_map
# convert_to_grayscale
# calculate_ssim

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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def get_color_map(im):
    return im / (rgb2gray(im)[..., np.newaxis] + 1e-6) * 100
    # return im / (np.mean(im, axis=-1)[..., np.newaxis] + 1e-6) * 100


def convert_to_grayscale(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def calculate_ssim(img1, img2):
    score, _ = SSIM(img1, img2, full=True)
    return score