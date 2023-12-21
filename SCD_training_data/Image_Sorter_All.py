import os
import glob
import random
import shutil

import tqdm

# C:\Users\Muroyama lab\Documents\Muroyama_Lab\Gabriel\GitHub\PDA-Acquisition\SCD_training_data\Train_stomata

DIR_OUT = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\full_training_set"

DIR_IN = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\test"

#DIR_OUT = "C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\SCD_training_data\\example_training_set"

#DIR_IN = "C:\\Users\\gjang\\Documents\\GitHub\\PDA-Acquisition\\source_images\\generated\\test"




# def randfile(dir):
#     assert os.path.exists(dir)
#     file_list = glob.glob(dir + "\\**")

#     #print(file_list)
#     return random.choice(file_list)

#print(randfile(DIR_IN + "\\base\\COT1_1648y-1193x\\*"))

def copy(src, dst):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    shutil.copyfile(src, dst)

#assert os.path.exists(DIR_IN + "\\base"), f"{DIR_IN}\\base"
num_files = len(glob.glob(DIR_IN + "\\base\\**\\*.png"))

train_split = 0.8
val_split = 0.1
test_split = 0.1

files_list = glob.glob(os.path.join(DIR_IN, "base\\**\\*.png"))


#print(folders_list)
random.shuffle(files_list)

train_index = round(train_split*num_files)
val_index = round(val_split*num_files)+train_index
#test_index = round(test_split*len(filenames))+val_index

train_list = files_list[:train_index]
val_list = files_list[train_index:val_index]
test_list = files_list[val_index:]

def movin(list, base_folder, anno_folder):
    for file in tqdm.tqdm(list):
        #print(folder)
        filename_base = file

        filename_anno_stage = filename_base.split("\\")
        #print(filename_anno_stage)
        filename_anno_stage[-3] = "anno"
        filename_anno = "\\".join(filename_anno_stage)

        copy(filename_base, os.path.join(DIR_OUT, base_folder))
        copy(filename_anno, os.path.join(DIR_OUT, anno_folder))


        #print(filename_anno)

# for folder in glob.glob(os.path.join(DIR_IN, "base\\**")):
#     filename_base = randfile(folder)
#     filename_anno_stage = filename_base.split("\\")
#     filename_anno_stage[12] = "anno"
#     filename_anno = "\\".join(filename_anno_stage)

#     copy(filename_base, os.path.join(DIR_OUT, "base"))
#     copy(filename_anno, DIR_OUT + "\\anno")


#     print(filename_anno)

movin(train_list, "train", "trainannot")
movin(val_list, "val", "valannot")
movin(test_list, "test", "testannot")





    #file = randfile(DIR_IN + "\\" + folder + "\\**")

#     print(folder)

# print(glob.glob(DIR_IN + "\\base\\**"))