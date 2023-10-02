from random import randint
from PIL import Image
from tifffile import imread, imwrite
import os
import numpy as np
from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import time

DIR_CWD = os.path.join(os.getcwd(), "SCD_training_data")
DIR_SOURCE = os.path.join(DIR_CWD, "source_images")

def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function{func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

SEARCH_FILE = imread("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\BASE\\cot1.tif")
MASK_FILE = imread("C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\ANNOTATION\\cot1_STOMATA_MASKS.tiff")

print(SEARCH_FILE.shape)
height, width = SEARCH_FILE.shape[0], SEARCH_FILE.shape[1]

segment_size = 64

@timer_func
def timeTaker(print_it = False):
    global SEARCH_FILE
    global MASK_FILE
    section_counts = [0,0,0]
    print(list(range(width))[-1])
    print(list(range(width))[0])
    for x in list(range(width)):
        for y in list(range(height)):
            chunk = SEARCH_FILE[y:y+segment_size,
                                x:x+segment_size]
            if chunk.size != 4096:
                break
            #print(chunk.size)
            if np.all(chunk[0, :] == 0) and np.all(chunk[:, 0] == 0) and np.all(chunk[:, segment_size-1] == 0) and np.all(chunk[segment_size-1, :] == 0):
                pass
            else:
                if np.mean(MASK_FILE[y:y+segment_size, x:x+segment_size]) > 0:
                    #section_counts[2] += 1
                    imwrite(f"C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\test\\COT1 - {x}x - {y}y.tif", chunk)

                            
    print(f'Empty Sections:{section_counts[0]}\nNon-Empty Sections:{section_counts[1]}\nAnnotated Sections:{section_counts[2]}')


timeTaker(print_it = True)