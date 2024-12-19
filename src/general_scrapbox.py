import csv
import glob
import os

import subprocess

from datetime import datetime, timedelta

import pandas as pd

import tqdm

# print(glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/BASE/*.tif"))
# print(glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/BASE/*"))

# with open('C:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\splits\\pores_only.csv', 'w', newline='', encoding='utf-8') as csvfile:

#     # for file in [x for x in glob.glob("C:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\only_pored\\BASE\\*.tif")]:
#     for file in [x for x in glob.glob("C:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\only_pored\\BASE\\*")]:
#         #print(file)
#         #print("C:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\only_pored\\ANNOT\\" + os.path.basename(file)[:-4] + "_ANNOT.png")
#         #print(os.path.exists(f"C:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\only_pored\\ANNOT\\" + os.path.basename(file)[:-4] + "_ANNOT.png"))
#         if os.path.exists(f"C:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\only_pored\\ANNOT\\" + os.path.basename(file)[:-4] + "_ANNOT.png"):
#             #print([os.path.basename(file), os.path.basename(file)[:-4] + "_ANNOT.png"])
            
#             rowwriter = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#             rowwriter.writerow([f"{file.strip(' ')},", 
#                                 f"C:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\only_pored\\ANNOT\\{os.path.basename(file)[:-4]}_ANNOT.png"
#                                 ])
            #print("**NO COUNTERPART**")
        
        #print(os.path.basename(file)[:-4])
# size = None
# time_0 = datetime.now()
# print("testing time")
# for file in [x for x in glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/AZD_test/base_images/*")]:
#     print(os.stat(file)[6]/(1024**2))
# time_1 = datetime.now()
# print(f"time: {time_1-time_0}")

# for file in glob.glob("C:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\only_pored\\ANNOT\\*.png"):
#     os.rename(file, file[:-3]+"pdn")

# SPEED size = 0
# SPEED time_0 = datetime.now()

# # for file in [x for x in glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/AZD_test/base_images_png/*")]:
# for file in [x for x in glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/AZD_test/slides_data/base/*")]:
#     #print(os.path.basename(file))
#     # SPEED size += os.stat(file)[6]/(1024**2)
#     # SPEED subprocess.run(f"python inference.py --model=checkpoints/2024-10-14_15h-07m-29s/last.e004.pth --input=only_pored/AZD_test/base_images_png/{os.path.basename(file)} --overlap=32 --outputname=_AZD_TEST_32_ --outputdir=only_pored/AZD_test/inference/", shell=True)
#     # subprocess.run(f"python inference.py --model=checkpoints/2024-10-14_15h-07m-29s/last.e004.pth --input=only_pored/AZD_test/base_images/{os.path.basename(file)} --overlap=48 --outputname=_AZD_TEST_48_ --outputdir=only_pored/AZD_test/inference/", shell=True)
#     filehandle = os.path.basename(file)
#     subprocess.run(f"python inference.py --model=checkpoints/2024-10-14_15h-07m-29s/last.e029.pth --input=only_pored/AZD_test/slides_data/base/{filehandle} --overlap=32 --outputname=_SLIDE_ --outputdir=only_pored/AZD_test/slides_data/inference/", shell=True)
#     subprocess.run(f"python errorviz.py --ground_truth=only_pored/AZD_test/slides_data/annot/{filehandle[:-4]}_ANNOT.png --model_predict=only_pored/AZD_test/slides_data/inference/{filehandle}_SLIDE_.output.png --save=only_pored/AZD_test/slides_data/{filehandle[:-4]}_DIFF.png", shell=True)
# SPEED time_1 = datetime.now()

# SPEED print(time_1-time_0)
# SPEED print(size)

# # for i in range(30):
# #     subprocess.run(f"python inference.py --model=checkpoints/2024-10-14_15h-07m-29s/last.e{i:03d}.pth --input=only_pored/AZD_test/base_images/{os.path.basename(file)} --overlap=48 --outputname=_AZD_TEST_48_{i:03d}_ --outputdir=only_pored/AZD_test/model_checker/", shell=True)

import data
import torch
import torchvision


# if __name__ == '__main__':
#     print("scround")
#     dummy_dataset = "SCD_training_data/mbn_training/classes/"
#     test_mbn_dataloader = data.create_dataloader_mbn(
#         ds = torchvision.datasets.ImageFolder(dummy_dataset, transform=data.to_tensor),
#         batchsize = 16,
#         num_workers = 2,
#         shuffle=True
#     )

#     print(isinstance(test_mbn_dataloader, torch.utils.data.DataLoader))


#     print(str(test_mbn_dataloader))
#     for i,[x,l] in enumerate(test_mbn_dataloader):
#         images, labels = x, l
#         print(f"Batch Shape: {images.shape}")
#         print(f"Labels: {labels}")

annotpath = "only_pored/ANNOT"
basepath = "only_pored/BASE"


# annotset = set([os.path.basename(i) for i in os.listdir(annotpath)])
# baseset = set([os.path.basename(i) for i in os.listdir(basepath)])

# diff = list(annotset-baseset)

# for d in diff:
#     a = True
#     b = True
#     if d not in annotset:
#         a = False
#         print(f"{d} is not present in annotset")
#     if d not in baseset:
#         assert d in annotset
#         b = False
#         print(f"{d} is not present in baseset")
#     assert a or b, "Welp"

filelist = [os.path.abspath(file) for file in glob.glob(os.path.join(annotpath, "*"))]

# current_csv = pd.read_csv("splits/pores_only.csv")
# current_set = set([row.iloc[1].strip(" ") for index, row in current_csv.iterrows()])
# filelist_set = set(filelist)

# remain = filelist_set-current_set
# counterpart = [f.replace("_ANNOT", "").replace("ANNOT", "BASE").replace("png", "tif") for f in remain if os.path.exists(f.replace("_ANNOT", "").replace("ANNOT", "BASE").replace("png", "tif"))]

# [print(f.replace("_ANNOT", "").replace("ANNOT", "BASE")) for f in remain]

# [print(f) for f in counterpart]
# assert len(counterpart) == len(remain), f"len counterpart = {len(counterpart)}, while len remain = {len(remain)}"

# TODO: Update splits!


# for f in tqdm.tqdm(filelist):
#     subprocess.run(f'python.exe clumps_table.py --input_path="{f}" --output_folder="inference/count_tables/" --prediction_type="clumps" --filter_type="otsu"', shell=True)
    #print("Analyzed ", f)


# # stomata counter
# amount = 0
# for t in os.listdir("inference/count_tables/"):
#     iter_csv = pd.read_csv("inference/count_tables/"+t)
#     print(t, ":\t\t\t", f"{len(iter_csv)}")
#     amount += len(iter_csv)
# print("Total to annotate: ", amount)

import cv2
import numpy as np

def tile_and_blur(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Create a 288x288 blank canvas
    canvas = np.zeros((288, 288, 3), dtype=np.uint8)

    # Define kernel sizes and sigmas
    kernel_sizes = [1, 3, 5, 7]
    sigmas = [0, 1, 2, 3]

    # Tile and blur
    for i in range(4):
        for j in range(4):
            # Create a copy of the image
            tile = img.copy()

            # Apply Gaussian Blur
            tile = cv2.GaussianBlur(tile, (kernel_sizes[j], kernel_sizes[j]), sigmas[i])

            # Resize the tile (if necessary)
            # tile = cv2.resize(tile, (72, 72))

            # Paste the tile onto the canvas
            x_offset = 72 * j
            y_offset = 72 * i
            canvas[y_offset:y_offset+72, x_offset:x_offset+72] = tile

    # Display or save the result
    cv2.imshow("Tiled and Blurred Image", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image
    cv2.imwrite("tiled_blurred_image.jpg", canvas)

# Example usage
image_path = "SCD_training_data/mbn_training/Untitled.png"
tile_and_blur(image_path)