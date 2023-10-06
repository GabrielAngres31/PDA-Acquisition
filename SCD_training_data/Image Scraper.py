from random import randint
from PIL import Image
from tifffile import imread, imwrite
import os
import numpy as np
from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import time

DIR_CWD = os.getcwd()
DIR_SOURCE = os.path.join(DIR_CWD, "source_images")
DIR_BASE = os.path.join(DIR_SOURCE, "BASE")
DIR_ANNO = os.path.join(DIR_SOURCE, "ANNOTATION")

def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function{func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

#SEARCH_FILE = imread(os.path.join(DIR_SOURCE, "BASE\\cot1.tif"))
#MASK_FILE = imread(os.path.join(DIR_SOURCE, "ANNOTATION\\cot1_STOMATA_MASKS.tiff"))

@timer_func
def timeTaker(searchfile, maskedfile, number, print_it = False):
    SEARCH_FILE = imread(searchfile)
    MASKED_FILE = imread(maskedfile)

    print(MASKED_FILE.shape)
    if MASKED_FILE.shape[2] == 4:
        
        MASKED_FILE = MASKED_FILE[:,:,:3]

    assert MASKED_FILE.shape[2] == 3
    print(MASKED_FILE.shape)
    
    height, width = SEARCH_FILE.shape[0], SEARCH_FILE.shape[1]
    segment_size = 64
    trial = 0
    section_counts = [0,0,0]
    sample_interval = 8
    
    for x in list(range(0, width, sample_interval)):
        for y in list(range(0, height, sample_interval)):
            #if trial == 50000:
            #    break
            #print(trial)
            chunk = SEARCH_FILE[y:y+segment_size,
                                x:x+segment_size]
            #print(chunk.size)
            #print(chunk.size)
            if chunk.size != 4096:
                break
            #print(chunk.size)
            #if np.all(chunk[0, :] == 0) and np.all(chunk[:, 0] == 0) and np.all(chunk[:, segment_size-1] == 0) and np.all(chunk[segment_size-1, :] == 0):
            #    pass
            #else:
            chunk_mask = MASKED_FILE[y:y+segment_size, 
                                     x:x+segment_size]
            label_string = ""
            # TODO: TROUBLESHOOT THIS!!!
            #print(f"{x} {y}")
            if np.all(chunk_mask == 0): # If no annotation is present in the cell
                #print("for")
                #print(chunk_mask)
                #print("-----")
                #print(chunk)
                #print(np.all(chunk<12))
                #assert 1 == 0
                if np.all(chunk < 12): # If the cell is mostly or all black, hop out of the loop
                    #print(f"INVALID #{trial}")
                    pass
                else: # Else, mark it as ABSENT (plant part that has no stomatal features)
                    label_string = "ABSENT"
            elif np.all(chunk_mask[0, :] == 0) and np.all(chunk_mask[:, 0] == 0) and np.all(chunk_mask[:, segment_size-1] == 0) and np.all(chunk_mask[segment_size-1, :] == 0):
                #print("ever")
                #print(chunk_mask[0, :])
                #print(chunk_mask[:, 0])
                #print(chunk_mask[:, segment_size-1])
                #print(chunk_mask[segment_size-1, :])
                # If annotation is present in the cell and there's no annotation content on the cell borders, then a stomata is wholly enclosed
                label_string = "WHOLE"
            else: # Otherwise it is not.
                #assert 1 == 0
                label_string = "PARTIAL"
            if label_string:
                imwrite(os.path.join(DIR_SOURCE, f"generated\\{label_string}\\COT{number}_{x}x_{y}y.png"), chunk)
            trial += 1
            if trial % 4000 == 0:
                print(f"{trial/1000}k images stored!")
            #print(f"Loop #{trial}")


    print(f"{trial/1000}k images stored!")
    #print(f'Empty Sections:{section_counts[0]}\nNon-Empty Sections:{section_counts[1]}\nAnnotated Sections:{section_counts[2]}')

def file_iterator():

    for i in list(range(1, 5)):
        searchfile_name = os.path.join(DIR_BASE, f"cot{i}.tif")
        maskedfile_name = os.path.join(DIR_ANNO, f"cot{i}_STOMATA_MASKS.tiff")
        timeTaker(searchfile_name, maskedfile_name, i, print_it = True)
    print("There was an attempt!")

file_iterator()