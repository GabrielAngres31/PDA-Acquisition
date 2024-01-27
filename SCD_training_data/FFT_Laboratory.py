# Fourier Transform Lab

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.color import rgb2gray

IMGS_TEST_DIR = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\tiles"
IMGS_TEST_DIR_ALT = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\FFT_OUTS\\Scrapbox"
IMG_TARGET = "cot1_(1002, 1680).tif"
IMG_TARGET_ALT = "Stomata_arms.tiff"

IMG_DIR = os.path.join(IMGS_TEST_DIR, IMG_TARGET)



def fft_transform(arr):

    s0 = rgb2gray(arr)
    #print("s1")
    s1 = np.fft.fft2(s0)
    #print("s2")
    s2 = np.fft.fftshift(s1)
    #print("s3")
    s3 = abs(s2)
    #print("s4")
    #print(s3)
    s4 = np.log(s3)
    #print("RETURN")
    s4[s4 == -np.inf] = 0

    # s5 = 
    return s4



IMAGE = cv2.imread(IMG_DIR)

FFT_IMAGE = fft_transform(IMAGE)

# for element in FFT_IMAGE:
#     print(element)
#     print("okay")

# FFT_IMAGE_2D = [[y[1] for y in x] for x in FFT_IMAGE]

# for element in FFT_IMAGE:
#     print(element)
#     print("okay")

def normalize(array_2d):
    max = np.max(array_2d)
    return np.multiply(np.divide(array_2d, max), 255)

# plt.figure(1)
# plt.subplot(121)
# plt.imshow(normalize(FFT_IMAGE), cmap='gray', vmin=0, vmax=255)
# plt.subplot(122)
# plt.imshow(normalize(fft_transform(np.subtract(255, IMAGE))), cmap='gray', vmin=0, vmax=255)

# plt.show()

if __name__ == "__main__":
    
    pass