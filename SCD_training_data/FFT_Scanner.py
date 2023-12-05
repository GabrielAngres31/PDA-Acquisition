import numpy as np
import matplotlib.pyplot as plt
import os

import glob

from skimage.io import imread, imshow, imsave

IMG_FILE_DIR = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\base"
IMGPATH_LIST = [file for file in glob.glob(f"{IMG_FILE_DIR}\\*.tif")]

print(IMGPATH_LIST)

def fft_transform(arr):
    return np.log(abs(np.fft.fftshift(np.fft.fft2(arr))))

def ITER_IMG(img):

    return []

