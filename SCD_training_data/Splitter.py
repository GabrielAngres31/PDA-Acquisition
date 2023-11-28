"""
Splitter.py
"""


import os
import numpy as np
import shutil
import random
import tqdm


filenames = os.listdir("source_images\\generated\\test\\base")
print(len(filenames))

train_split = 0.8
val_split = 0.1
test_split = 0.1

random.shuffle(filenames)

train_index = round(train_split*len(filenames))
val_index = round(val_split*len(filenames))+train_index
#test_index = round(test_split*len(filenames))+val_index

train_list = filenames[:train_index]
val_list = filenames[train_index:val_index]
test_list = filenames[val_index:]


assert len(train_list) + len(val_list) + len(test_list) == len(filenames), f"{len(train_list)} + {len(val_list)} + {len(test_list)} =/= {len(filenames)}"

DIR_SOURCE = "source_images\\generated\\test"
DIR_S_ANNO = os.path.join(DIR_SOURCE, "anno")
DIR_S_BASE = os.path.join(DIR_SOURCE, "base")

DIR__OUTPUT = "Train_stomata\\data"
DIR_O__TEST = os.path.join(DIR__OUTPUT, "test") 
DIR_O___VAL = os.path.join(DIR__OUTPUT, "val")
DIR_O_TRAIN = os.path.join(DIR__OUTPUT, "train")




for i in tqdm.tqdm(train_list):
    shutil.copyfile(DIR_S_BASE + "\\"+ i, DIR_O_TRAIN + "\\" + i)
    shutil.copyfile(DIR_S_ANNO + "\\"+ i, DIR_O_TRAIN + "annot" + "\\" + i)

for i in tqdm.tqdm(val_list):
    shutil.copyfile(DIR_S_BASE + "\\"+ i, DIR_O___VAL + "\\" + i)
    shutil.copyfile(DIR_S_ANNO + "\\"+ i, DIR_O___VAL + "annot" + "\\" + i)

for i in tqdm.tqdm(test_list):
    shutil.copyfile(DIR_S_BASE + "\\"+ i, DIR_O__TEST + "\\" + i)
    shutil.copyfile(DIR_S_ANNO + "\\"+ i, DIR_O__TEST + "annot" + "\\" + i)