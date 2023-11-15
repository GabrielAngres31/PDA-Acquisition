from random import randint
from PIL import Image
from tifffile import imread, imwrite
import os
import numpy as np
from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import time

import tqdm

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
    
    def chunksearch(x, y, seg_size):
        #if trial == 50000:
            #    break
            #print(trial)
            chunk = SEARCH_FILE[y:y+seg_size,
                                x:x+seg_size]
            #print(chunk.size)
            #print(chunk.size)
            #if chunk.size != 4096:
            #    break
            #print(chunk.size)
            #if np.all(chunk[0, :] == 0) and np.all(chunk[:, 0] == 0) and np.all(chunk[:, segment_size-1] == 0) and np.all(chunk[segment_size-1, :] == 0):
            #    pass
            #else:
            chunk_mask = MASKED_FILE[y:y+seg_size, 
                                     x:x+seg_size]
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
            elif np.all(chunk_mask[0, :] == 0) and np.all(chunk_mask[:, 0] == 0) and np.all(chunk_mask[:, seg_size-1] == 0) and np.all(chunk_mask[seg_size-1, :] == 0):
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
                imwrite(os.path.join(DIR_SOURCE, f"Trimmed\\generated\\COT{number}_trim_{x}x_{y}y.png"), chunk)
            trial += 1
            if trial % 4000 == 0:
                print(f"{trial/1000}k images stored!")
            #print(f"Loop #{trial}")
            
            print(f"{trial/1000}k images stored!")

    chunkgenerator = (chunksearch(x, y, segment_size) for x in range(0, width, sample_interval) for y in range(0, height, sample_interval))


    #print(f'Empty Sections:{section_counts[0]}\nNon-Empty Sections:{section_counts[1]}\nAnnotated Sections:{section_counts[2]}')



def updatedScraper(searchfile, annofile, number):

    SEARCH_FILE = imread(searchfile)
    ANNO_FILE = imread(annofile)
    def chunk_getter(x, y, super_seg_size):
        
        #anno_chunk = ANNO_FILE[y+1:y+super_seg_size-1,
        #                    x+1:x+super_seg_size-1]

        anno_superchunk = ANNO_FILE[y:y+super_seg_size,
                            x:x+super_seg_size].astype(bool)
        
        anno_super_ring = np.copy(anno_superchunk)
        anno_super_ring[1:-1, 1:-1] = 0

        if np.count_nonzero(anno_superchunk) == 0:
            return None

        if np.count_nonzero(anno_super_ring)  > 0:
            return None

        #if not(base_chunk.shape[0] == base_chunk.shape[1] == super_seg_size-2):
        if not(anno_superchunk.shape[0] == anno_superchunk.shape[1] == super_seg_size):
            return None
        
        # if np.count_nonzero(anno_super_ring) == 0:
        #     return None
        
        base_chunk = SEARCH_FILE[y+1:y+super_seg_size-1,
                        x+1:x+super_seg_size-1]
        
        anno_chunk = ANNO_FILE[y+1:y+super_seg_size-1,
                        x+1:x+super_seg_size-1]

        return (base_chunk, anno_chunk, x, y)
        
        # if base_chunk.shape[0] == base_chunk.shape[1] == super_seg_size-2:
        #     if np.count_nonzero(anno_chunk)-np.count_nonzero(anno_superchunk):
        #         if np.any(base_chunk > 32):
                    
        #             print("WORKED")
        #             return base_chunk
        # else:
        #     val = np.count_nonzero(anno_chunk)-np.count_nonzero(anno_superchunk)
        #     # if val != 0: 
        #     #     print(val)
        #     #     print("-----")
        #     return None
    # 72x72
    
    # TODO: WHY THE FUCK IS THIS WRITING EVERYTHING INSTEAD OF JSUT THE STOMATA
    return (chunk_getter(x, y, 64+2) for x in range(0, SEARCH_FILE.shape[1]) for y in range(0, SEARCH_FILE.shape[0]))


def file_iterator():
    num = 0

    for i in tqdm.tqdm(range(1, 5)):
        searchfile_name = os.path.join(DIR_BASE, f"cot{i}.tif")
        annofile_name =   os.path.join(DIR_ANNO, f"cot{i}_STOMATA_MASKS.tiff")
        #maskedfile_name = os.path.join(DIR_ANNO, f"cot{i}_STOMATA_MASKS.tiff")
        def dual_write(path0, file0, path1, file1):
            imwrite(path0, file0)
            imwrite(path1, file1)  
        [
            dual_write(os.path.join(DIR_SOURCE, "generated\\test\\base", f"COT{i}_{unit[3]}y-{unit[2]}x.png"), unit[0], 
                       os.path.join(DIR_SOURCE, "generated\\test\\anno", f"COT{i}_{unit[3]}y-{unit[2]}x.png"), unit[1])
                       for unit in updatedScraper(searchfile_name, annofile_name, i) if unit is not None
                       ]
        #[imwrite(os.path.join(DIR_SOURCE, "generated\\test", f"COT{i}_trim_{unit[3]}y-{unit[2]}x.png"), unit[0]) for unit in updatedScraper(searchfile_name, annofile_name, i) if unit is not None]  #maskedfile_name,
    print("There was an attempt!")

file_iterator()