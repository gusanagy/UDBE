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

def plot_images(images):
    n = len(images)
    # Calcula o layout da grade para as imagens
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis('off')  # Desliga os eixos
    
    # Desliga eixos para os subplots não usados
    for ax in axes[n:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
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


def low_light_adjust(mean,std,lower_bound=0.2):
    """
    Adjust the mean value to ensure that the lower bound of the light distribution 
    does not fall below a specified threshold.

    This function modifies the mean value to ensure that the lower bound, 
    defined as `mean - 2 * std`, is not less than the specified `lower_bound`. 
    It iteratively increases the mean value until the condition is met.

    Parameters:
    mean (float): The mean value of the light distribution.
    std (float): The standard deviation of the light distribution.
    lower_bound (float, optional): The minimum acceptable lower bound for the 
                                   light distribution. Default is 0.2.

    Returns:
    tuple: A tuple containing the adjusted lower bounds of the light distribution 
           (mean - 2 * std, mean - std).
    """
    while mean - std*2 <lower_bound: mean += 0.2;# print("mean low light",mean)    

    return (mean - std*2, mean-std)

def high_light_adjust(mean,std,high_bound=1.6, lower_bound=1.0):
    """
    Adjust the mean value to ensure that the high and low bounds of the light distribution 
    fall within specified thresholds.

    This function modifies the mean value to ensure that the high bound, 
    defined as `4 * std + mean`, does not exceed the specified `high_bound`, 
    and the low bound, defined as `2 * std + mean`, does not fall below the specified 
    `lower_bound`. It iteratively adjusts the mean value until both conditions are met.

    Parameters:
    mean (float): The mean value of the light distribution.
    std (float): The standard deviation of the light distribution.
    high_bound (float, optional): The maximum acceptable high bound for the 
                                  light distribution. Default is 1.6.
    lower_bound (float, optional): The minimum acceptable low bound for the 
                                   light distribution. Default is 1.0.

    Returns:
    tuple: A tuple containing the adjusted bounds of the light distribution 
           (2 * std + mean, 4 * std + mean).
    """
    while 4*std+mean > high_bound: mean-=0.1;#print("mean high light", mean)
    while 2*std+mean < lower_bound: mean+=0.1

    return (2*std+mean, 4*std+mean)