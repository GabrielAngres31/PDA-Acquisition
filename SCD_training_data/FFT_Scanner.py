import numpy as np
import matplotlib.pyplot as plt
import os

import glob

from skimage.io import imread, imshow, imsave

from collections import namedtuple

IMG_FILE_DIR = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE"
IMG_FILE_SAVE = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\FFT_OUTS"
IMGPATH_LIST = ["COT1.tif", "COT2.tif", "COT3.tif", "COT4.tif", "COT5.tif", "COT6.tif"]

print(IMGPATH_LIST)

counter = 0

def fft_transform(arr):
    #global counter
    #counter += 1
    #print("s1")
    s1 = np.fft.fft2(arr)
    #print("s2")
    s2 = np.fft.fftshift(s1)
    #print("s3")
    s3 = abs(s2)
    #print("s4")
    #print(s3)
    s4 = np.log(s3)
    #print("RETURN")
    #if counter % 5000 == 0: print(counter)
    return s4

SHAPE = namedtuple("SHAPE", "x_width y_height")

def ITER_IMG(img, chunksize = SHAPE(64, 64)):
    img_shape = SHAPE(img.shape[1], img.shape[0])
    limits = SHAPE(img_shape.x_width - chunksize.x_width, img_shape.y_height - chunksize.y_height)

    def chunkfinder(image, x, y, x_seg, y_seg):
        return image[y:y+y_seg, x:x+x_seg]

    #return [(chunkfinder(img, x_c, y_c, chunksize.x_width, chunksize.y_height), x_c, y_c) for x_c in range(limits.x_width) for y_c in range(limits.y_height)]
    return [chunkfinder(img, x_c, y_c, chunksize.x_width, chunksize.y_height) for x_c in range(limits.x_width) for y_c in range(limits.y_height)]

def ITER_FFT(ITER_OUT):
    return [fft_transform(img) for img in ITER_OUT]


def fft_SPLITTER(chunksize, main_image_filetitle):
    
    print(f"Reading {main_image_filetitle} into memory...")
    main_image = imread(os.path.join(IMG_FILE_DIR, main_image_filetitle))

    img_shape = SHAPE(main_image.shape[1], main_image.shape[0])
    limits = SHAPE(img_shape.x_width - chunksize.x_width, img_shape.y_height - chunksize.y_height)

    print("Generating image segments...")
    ITER = ITER_IMG(main_image, chunksize)

    print(f"Conducting Fourier transforms on image segments...")
    FFT = ITER_FFT(ITER)
    FFT_compact = np.array(FFT)

    COLLECT = namedtuple("COLLECT", "long x y")

    print("Splitting into component frequencies (approximate)...")
    FFT_split = [COLLECT(FFT_compact[:, x_c, y_c], x_c, y_c) for x_c in range(chunksize.x_width) for y_c in range(chunksize.y_height)]

    print(f"Beginning {main_image_filetitle} storage!...")
    return [imsave(os.path.join(IMG_FILE_SAVE, f"{main_image_filetitle[:-4]}_{fft_long.x}x-{fft_long.y}y.jpg"), fft_long.reshape((limits.y_height, limits.x_width))) for fft_long in FFT_split]
    # storepath = os.path.join(IMG_FILE_SAVE, f"")
    # return [imsave()]

    pass

[fft_SPLITTER(SHAPE(64, 64), img_title) for img_title in IMGPATH_LIST]

