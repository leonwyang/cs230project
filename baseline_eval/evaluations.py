import numpy as np
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
from PIL import Image

    

MAX_PIX = 255
def MSE(img1, img2):
    return np.mean( (img1-img2)**2)

def PSNR(img1,img2):
    mse = MSE(img1, img2)
    return 10 * np.log10(MAX_PIX**2/mse)

def SSIM_Array(arr1, arr2):
    img1 = Image.fromarray(arr1.astype('uint8'), 'RGB')
    img2 = Image.fromarray(arr2.astype('uint8'),'RGB')
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)
    return SSIM(img1,gaussian_kernel_1d).ssim_value(img2)
