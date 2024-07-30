
#https://github.com/xueleichen/PSNR-SSIM-UCIQE-UIQM-Python/blob/main/nevaluate.py
'''
Metrics for unferwater image quality evaluation.

Author: Xuelei Chen 
Email: chenxuelei@hotmail.com

Usage:
python evaluate.py RESULT_PATH
'''
#calcular UQIM
#calcular UCIQE
import numpy as np
#from skimage.measure import compare_psnr, compare_ssim
import math
import sys
from skimage import io, color, filters
import os
import math
#import wandb
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM 
from scipy import ndimage
from PIL import Image
import numpy as np
import math


import cv2
import numpy as np


def uciqe(nargin,loc):
    #img_bgr = cv2.imread(loc)        # Used to read image files
     
    img_lab = cv2.cvtColor(loc, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    if nargin == 1:                                 # According to training result mentioned in the paper:
        coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    quality_val = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    # print("quality_val is", quality_val)
    return quality_val
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    """
      Calculates the asymetric alpha-trimmed mean
    """
    # sort pixels by intensity - for clipping
    x = sorted(x)
    # get number of pixels
    K = len(x)
    # calculate T alpha L and T alpha R
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    # calculate mu_alpha weight
    weight = (1 / (K - T_a_L - T_a_R))
    # loop through flattened image starting at T_a_L+1 and ending at K-T_a_R
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    val = sum(x[s:e])
    val = weight * val
    return val


def s_a(x, mu):
    val = 0
    for pixel in x:
        val += math.pow((pixel - mu), 2)
    return val / len(x)


def _uicm(x):
    R = x[:, :, 0].flatten()
    G = x[:, :, 1].flatten()
    B = x[:, :, 2].flatten()
    RG = R - G
    YB = ((R + G) / 2) - B
    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)
    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)
    l = math.sqrt((math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2)))
    r = math.sqrt(s_a_RG + s_a_YB)
    return (-0.0268 * l) + (0.1586 * r)


def sobel(x):
    dx = ndimage.sobel(x, 0)
    dy = ndimage.sobel(x, 1)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.max(mag)
    return mag


def eme(x, window_size):
    """
      Enhancement measure estimation
      x.shape[0] = height
      x.shape[1] = width
    """
    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size

    # weight
    w = 2. / (k1 * k2)

    blocksize_x = window_size
    blocksize_y = window_size

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y * k2, :blocksize_x * k1]

    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1)]
            max_ = np.max(block)
            min_ = np.min(block)

            # bound checks, can't do log(0)
            if min_ == 0.0:
                val += 0
            elif max_ == 0.0:
                val += 0
            else:
                val += math.log(max_ / min_)
    return w * val


def _uism(x):
    """
      Underwater Image Sharpness Measure
    """
    # get image channels
    R = x[:, :, 0]
    G = x[:, :, 1]
    B = x[:, :, 2]

    # first apply Sobel edge detector to each RGB component
    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)

    # multiply the edges detected for each channel by the channel itself
    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)

    # get eme for each channel
    r_eme = eme(R_edge_map, 8)
    g_eme = eme(G_edge_map, 8)
    b_eme = eme(B_edge_map, 8)

    # coefficients
    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144

    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)


def plip_g(x, mu=1026.0):
    return mu - x


def plip_theta(g1, g2, k):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return k * ((g1 - g2) / (k - g2))


def plip_cross(g1, g2, gamma):
    g1 = plip_g(g1)
    g2 = plip_g(g2)
    return g1 + g2 - ((g1 * g2) / (gamma))


def plip_diag(c, g, gamma):
    g = plip_g(g)
    return gamma - (gamma * math.pow((1 - (g / gamma)), c))


def plip_multiplication(g1, g2):
    return plip_phiInverse(plip_phi(g1) * plip_phi(g2))
    # return plip_phiInverse(plip_phi(plip_g(g1)) * plip_phi(plip_g(g2)))


def plip_phiInverse(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return plip_lambda * (1 - math.pow(math.exp(-g / plip_lambda), 1 / plip_beta));


def plip_phi(g):
    plip_lambda = 1026.0
    plip_beta = 1.0
    return -plip_lambda * math.pow(math.log(1 - g / plip_lambda), plip_beta)


def _uiconm(x, window_size):
    """
      Underwater image contrast measure
      https://github.com/tkrahn108/UIQM/blob/master/src/uiconm.cpp
      https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5609219
    """

    plip_lambda = 1026.0
    plip_gamma = 1026.0
    plip_beta = 1.0
    plip_mu = 1026.0
    plip_k = 1026.0

    # if 4 blocks, then 2x2...etc.
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size

    # weight
    w = -1. / (k1 * k2)

    blocksize_x = window_size
    blocksize_y = window_size

    # make sure image is divisible by window_size - doesn't matter if we cut out some pixels
    x = x[:blocksize_y * k2, :blocksize_x * k1]

    # entropy scale - higher helps with randomness
    alpha = 1

    val = 0
    for l in range(k1):
        for k in range(k2):
            block = x[k * window_size:window_size * (k + 1), l * window_size:window_size * (l + 1), :]
            max_ = np.max(block)
            min_ = np.min(block)

            top = max_ - min_
            bot = max_ + min_

            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                val += alpha * math.pow((top / bot), alpha) * math.log(top / bot)

            # try: val += plip_multiplication((top/bot),math.log(top/bot))
    return w * val


def getUIQM(x):
    """
      Function to return UIQM to be called from other programs
      x: image
    """
    x = x.astype(np.float32)
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    # c1 = 0.4680; c2 = 0.2745; c3 = 0.2576
    ### from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7300447
    c1 = 0.0282
    c2 = 0.2953
    c3 = 3.5753

    uicm = _uicm(x)
    uism = _uism(x)
    uiconm = _uiconm(x, 8)
    uiqm = (c1 * uicm) + (c2 * uism) + (c3 * uiconm)
    return uiqm

def nmetrics(a):
    rgb = a
    lab = color.rgb2lab(a)
    gray = color.rgb2gray(a)
    # UCIQE
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    # c1 = 0.2745 # (menos peso, pois o brilho pode afetar a precisão da medição da cromaticidade)
    # c2 = 0.3742 # (maior peso, pois a saturação é importante para imagens com brilho)
    # c3 = 0.3743 # (maior peso, pois o contraste de luminância é crucial para imagens com brilho)
    l = lab[:,:,0]

    #1st term
    chroma = (lab[:,:,1]**2 + lab[:,:,2]**2)**0.5
    uc = np.mean(chroma)
    sc = (np.mean((chroma - uc)**2))**0.5

    #2nd term
    top = int(np.round(0.01 * l.shape[0] * l.shape[1]))
    #top = np.int(np.round(0.01*l.shape[0]*l.shape[1]))
    sl = np.sort(l,axis=None)
    isl = sl[::-1]
    conl = np.mean(isl[:top])-np.mean(sl[:top])

    #3rd term
    satur = []
    chroma1 = chroma.flatten()
    l1 = l.flatten()
    for i in range(len(l1)):
        if chroma1[i] == 0: satur.append(0)
        elif l1[i] == 0: satur.append(0)
        else: satur.append(chroma1[i] / l1[i])

    us = np.mean(satur)

    uciqe = c1 * sc + c2 * conl + c3 * us

    # UIQM
    p1 = 0.0282
    p2 = 0.2953
    p3 = 3.5753
    # p1 = 0.2907 # (menos peso, pois o brilho pode afetar a precisão da medição da cor)
    # p2 = 0.5155 # (maior peso, pois a nitidez é crucial para imagens com brilho)
    # p3 = 0.1938 # (peso moderado para o contraste)

    #1st term UICM
    rg = rgb[:,:,0] - rgb[:,:,1]
    yb = (rgb[:,:,0] + rgb[:,:,1]) / 2 - rgb[:,:,2]
    rgl = np.sort(rg,axis=None)
    ybl = np.sort(yb,axis=None)
    al1 = 0.1
    al2 = 0.1
    T1 = int(al1 * len(rgl))
    T2 = int(al2 * len(rgl))
    rgl_tr = rgl[T1:-T2]
    ybl_tr = ybl[T1:-T2]

    urg = np.mean(rgl_tr)
    s2rg = np.mean((rgl_tr - urg) ** 2)
    uyb = np.mean(ybl_tr)
    s2yb = np.mean((ybl_tr- uyb) ** 2)

    uicm =-0.0268 * np.sqrt(urg**2 + uyb**2) + 0.1586 * np.sqrt(s2rg + s2yb)

    #2nd term UISM (k1k2=8x8)
    Rsobel = rgb[:,:,0] * filters.sobel(rgb[:,:,0])
    Gsobel = rgb[:,:,1] * filters.sobel(rgb[:,:,1])
    Bsobel = rgb[:,:,2] * filters.sobel(rgb[:,:,2])

    Rsobel=np.round(Rsobel).astype(np.uint8)
    Gsobel=np.round(Gsobel).astype(np.uint8)
    Bsobel=np.round(Bsobel).astype(np.uint8)

    Reme = eme(Rsobel)
    Geme = eme(Gsobel)
    Beme = eme(Bsobel)

    uism = 0.299 * Reme + 0.587 * Geme + 0.114 * Beme

    #3rd term UIConM
    uiconm = logamee(gray)

    uiqm = p1 * uicm + p2 * uism + p3 * uiconm
    return uiqm,uciqe

def eme(ch,blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    eme = 0
    w = 2. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb,ylb:yrb]

            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            # # old version
            # if blockmin == 0.0: eme += 0
            # elif blockmax == 0.0: eme += 0
            # else: eme += w * math.log(blockmax / blockmin)

            # new version
            if blockmin == 0: blockmin+=1
            if blockmax == 0: blockmax+=1
            eme += w * math.log(blockmax / blockmin)
    return eme

def plipsum(i,j,gamma=1026):
    return i + j - i * j / gamma

def plipsub(i,j,k=1026):
    return k * (i - j) / (k - j)

def plipmult(c,j,gamma=1026):
    return gamma - gamma * (1 - j / gamma)**c

def logamee(ch, blocksize=8):

    num_x = math.ceil(ch.shape[0] / blocksize)
    num_y = math.ceil(ch.shape[1] / blocksize)
    
    s = 0
    w = 1. / (num_x * num_y)
    for i in range(num_x):

        xlb = i * blocksize
        if i < num_x - 1:
            xrb = (i+1) * blocksize
        else:
            xrb = ch.shape[0]

        for j in range(num_y):

            ylb = j * blocksize
            if j < num_y - 1:
                yrb = (j+1) * blocksize
            else:
                yrb = ch.shape[1]
            
            block = ch[xlb:xrb, ylb:yrb]
            blockmin = float(np.min(block))
            blockmax = float(np.max(block))

            top = plipsub(blockmax, blockmin)
            bottom = plipsum(blockmax, blockmin)

            if bottom == 0:
                m = 0
            else:
                m = top / bottom
            
            if m != 0:
                s += m * np.log(m)

    return plipmult(w, s)


def main():
    #dataset val // dataset inferencia // dataset // author
    avaliacao = [ 
        ("/home/gusanagy/Documents/Glown-Diffusion/data/UDWdata/UIEB/val","/home/gusanagy/Documents/Glown-Diffusion/data/UDWdata/UIEB/val", "Claudio", "UIEB")
    ]

    for candidato in avaliacao:
        result_path ,gt, author ,dataset = candidato
        print(f"author: {author} dataset: {dataset}")
        
        result_dirs = os.listdir(result_path)
        result_gt = os.listdir(gt)

        sumuiqm, sumuciqe = 0.,0.
        psnr_sum, ssim_sum = 0., 0.
        N=0

        for imgdir, gt_file in tqdm(zip(result_dirs, result_gt), total=len(result_dirs)):
            if '.png' in imgdir or '.jpg' in imgdir and '.png' in gt_file or '.jpg' in gt_file:
                try:
                    corrected = io.imread(os.path.join(result_path, imgdir))
                    #gt_image = io.imread(os.path.join(result_path,gt))
                    gt_image = io.imread(os.path.join(gt, gt_file)) #ALTEREI
                except Exception as e:
                    print(f"Erro ao carregar a imagem: {e}")
                    continue

                try:
                    #uiqm, uciqe = nmetrics(corrected)
                    uciqe_ = uciqe(nargin=1,loc=corrected)#usarei este
                    uiqm ,_ = nmetrics(corrected)#usarei o uiqm daqui

                    psnr_value = PSNR(gt_image, corrected) #, data_range=255)
                    ssim_value = SSIM(gt_image, corrected, multichannel=True, win_size=3) #, data_range=255)
                except Exception as e:
                    print(f"Erro ao calcular métricas: {e}")
                    continue

                sumuiqm += uiqm
                sumuciqe += uciqe_
                psnr_sum += psnr_value
                ssim_sum += ssim_value
                N += 1

        muiqm = sumuiqm / N
        muciqe = sumuciqe / N
        psnr_average = psnr_sum / N
        ssim_average = ssim_sum / N
        #print(avaliacao[0][:-1],avaliacao[0][1])
        print(f'Average: uiqm={muiqm} uciqe={muciqe} psnr = {psnr_average} ssim = {ssim_average}')
        file_path = os.path.join(avaliacao[0][1], avaliacao[0][:-1]+'metrics_output.txt')
        print(file_path)
        with open(file_path, 'w') as file:
            file.write(f'{avaliacao[0][:-1]}\nAverage: uiqm={muiqm}\n uciqe={muciqe}\n psnr={psnr_average}\n ssim={ssim_average}\n')

    print(f'Average: uiqm={muiqm} uciqe={muciqe} psnr={psnr_average} ssim={ssim_average}')
if __name__ == '__main__':
    main()