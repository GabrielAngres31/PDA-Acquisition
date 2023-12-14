import os
import glob
import random
import shutil


DIR_OUT = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\example_training_set"

DIR_IN = "C:\\Users\\Muroyama lab\\Documents\\Muroyama_Lab\\Gabriel\\GitHub\\PDA-Acquisition\\SCD_training_data\\source_images\\generated\\test"

def randfile(dir):
    file_list = glob.glob(dir + "\\**")

    #print(file_list)
    return  random.choice(file_list)

#print(randfile(DIR_IN + "\\base\\COT1_1648y-1193x\\*"))

def copy(src, dst):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    shutil.copyfile(src, dst)


num_folders = len(glob.glob(os.path.join(DIR_IN, "base\\**")))

train_split = 0.8
val_split = 0.1
test_split = 0.1


folders_list = glob.glob(os.path.join(DIR_IN, "base\\**"))
random.shuffle(folders_list)

train_index = round(train_split*num_folders)
val_index = round(val_split*num_folders)+train_index
#test_index = round(test_split*len(filenames))+val_index

train_list = folders_list[:train_index]
val_list = folders_list[train_index:val_index]
test_list = folders_list[val_index:]

def movin(list, base_folder, anno_folder):
    for folder in list:
        filename_base = randfile(folder)
        filename_anno_stage = filename_base.split("\\")
        filename_anno_stage[12] = "anno"
        filename_anno = "\\".join(filename_anno_stage)

        copy(filename_base, os.path.join(DIR_OUT, base_folder))
        copy(filename_anno, os.path.join(DIR_OUT, anno_folder))


        print(filename_anno)

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