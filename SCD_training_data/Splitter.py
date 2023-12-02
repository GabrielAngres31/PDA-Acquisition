"""
Splitter.py
"""


import os
import numpy as np
import shutil
import random
import tqdm



def pickRandomFolders(N, subdir):
    folders = os.listdir(subdir)
    random.shuffle(folders)
    return folders[:N]

def pickRandomImage(subfold):
    files = os.listdir("source_images\\generated\\test\\base\\"+ subfold)
    random.shuffle(files)
    return subfold + "\\"+ files[0]

# RETURN THE FULL FILE NAME, INCLUDING THE SUBFOLDER, OR YOU ARE S.O.L.


filenames = [pickRandomImage(sf) for sf in os.listdir("source_images\\generated\\test\\base")] #os.listdir("source_images\\generated\\test\\base")
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

def modulo_cutoff(list_in, mod):
    diff = len(list_in) % mod
    return list_in[diff:]

#train_list = modulo_cutoff(train_list, 8)
#val_list = modulo_cutoff(val_list, 8)
#test_list = modulo_cutoff(test_list, 8)

DIR_SOURCE = "source_images\\generated\\test"
DIR_S_ANNO = os.path.join(DIR_SOURCE, "anno")
DIR_S_BASE = os.path.join(DIR_SOURCE, "base")

DIR__OUTPUT = "Train_stomata"
DIR_O__TEST = os.path.join(DIR__OUTPUT, "test") 
DIR_O___VAL = os.path.join(DIR__OUTPUT, "val")
DIR_O_TRAIN = os.path.join(DIR__OUTPUT, "train")
DIR_O___ALL = os.path.join(DIR__OUTPUT, "all")


counter = 0

for ind, file in tqdm.tqdm(enumerate(train_list)):
    counter += 1
    shutil.copyfile(DIR_S_BASE + "\\"+ file, DIR_O_TRAIN + "\\" + str(ind) + "trn_" + "b.png")
    shutil.copyfile(DIR_S_ANNO + "\\"+ file, DIR_O_TRAIN + "annot" + "\\" + str(ind) + "trn_ann_" + "b.png")
    shutil.copyfile(DIR_S_BASE + "\\"+ file, DIR_O___ALL + "\\img_" + str(counter) + ".png")
    shutil.copyfile(DIR_S_ANNO + "\\"+ file, DIR_O___ALL + "annot" + "\\ann_img_" + str(counter) + ".png")
    

for ind, file in tqdm.tqdm(enumerate(val_list)):
    counter += 1
    shutil.copyfile(DIR_S_BASE + "\\"+ file, DIR_O___VAL + "\\" + str(ind) + "val_" + "b.png")
    shutil.copyfile(DIR_S_ANNO + "\\"+ file, DIR_O___VAL + "annot" + "\\" + str(ind) + "val_ann_" + "b.png")
    shutil.copyfile(DIR_S_BASE + "\\"+ file, DIR_O___ALL + "\\img_" + str(counter) + ".png")
    shutil.copyfile(DIR_S_ANNO + "\\"+ file, DIR_O___ALL + "annot" + "\\ann_img_" + str(counter) + ".png")


for ind, file in tqdm.tqdm(enumerate(test_list)):
    counter += 1
    shutil.copyfile(DIR_S_BASE + "\\"+ file, DIR_O__TEST + "\\" + str(ind) + "tst_" + "b.png")
    shutil.copyfile(DIR_S_ANNO + "\\"+ file, DIR_O__TEST + "annot" + "\\" + str(ind) + "tst_ann_" + "b.png")
    shutil.copyfile(DIR_S_BASE + "\\"+ file, DIR_O___ALL + "\\img_" + str(counter) + ".png")
    shutil.copyfile(DIR_S_ANNO + "\\"+ file, DIR_O___ALL + "annot" + "\\ann_img_" + str(counter) + ".png")

