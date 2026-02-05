import csv
import glob
import os

import subprocess

from datetime import datetime, timedelta

import pandas as pd

import tqdm
import shutil

from PIL import Image
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

# filelist = [os.path.abspath(file) for file in glob.glob(os.path.join(annotpath, "*"))]

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
# tile_and_blur(image_path)


import csv
import os
import shutil

def copy_files_from_csv(csv_file, destination_folder):
  """
  Reads filepaths from a CSV file and copies the corresponding files to a new folder.

  Args:
    csv_file: Path to the CSV file containing the filepaths.
    destination_folder: Path to the destination folder for the copied files.
  """

#   if not os.path.exists(destination_folder):
#     os.makedirs(destination_folder)

#   with open(csv_file, 'r') as f:
#     reader = csv.reader(f)
#     next(reader)  # Skip header row if present

#     for row in reader:
#       try:
#         for filepath in [row[0].strip(" "), row[1].strip(" ")]:

#             filename = os.path.basename(filepath)
#             destination_path = os.path.join(destination_folder, filename)

#             shutil.copy2(filepath, destination_path)
#             print(f"Copied {filepath} to {destination_path}")

#       except FileNotFoundError:
#         print(f"File not found: {filepath}")
#       except Exception as e:
#         print(f"Error copying {filepath}: {e}")

# if __name__ == "__main__":
#   csv_file = "splits/pores_only_val_12-2024.csv"  # Replace with the actual path to your CSV file
#   destination_folder = "to_zip_PDA_Acq_val_12-2024"  # Replace with the desired destination folder

#   copy_files_from_csv(csv_file, destination_folder)

# for i in range(1, 9):
#   print(i)
#   subprocess.run(f'python inference.py --model=checkpoints/2024-12-23_17h-07m-06s/last.e029.pth --input="only_pored/AZD_test/base_images/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_{i}_1_Merged.tif" --overlap=128 --outputname="_AZD_jan2025_1uMAZD_{i}_1"', shell=True)
# for i in range(1, 6):
#    print(i)
#    subprocess.run(f'python inference.py --model=checkpoints/2024-12-23_17h-07m-06s/last.e029.pth --input="only_pored/AZD_test/base_images/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_{i}_1_Merged.tif" --overlap=128 --outputname="_AZD_jan2025_100nMAZD_{i}_1"', shell=True)
#    subprocess.run(f'python inference.py --model=checkpoints/2024-12-23_17h-07m-06s/last.e029.pth --input="only_pored/AZD_test/base_images/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_{i}_1_Merged.tif" --overlap=128 --outputname="_AZD_jan2025_DMSO_{i}_1"', shell=True)
#    subprocess.run(f'python inference.py --model=checkpoints/2024-12-23_17h-07m-06s/last.e029.pth --input="only_pored/AZD_test/base_images/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_{i}_1_Merged.tif" --overlap=128 --outputname="_AZD_jan2025_250nMAZD_{i}_1"', shell=True)



# for i in range(1, 9):
#   subprocess.run(f"python clumps_table.py --input_path=inference/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_{i}_1_Merged.tif_AZD_jan2025_1uMAZD_{i}_1.output.png --prediction_type=clumps --filter_type=otsu --save_image_as=AZD_1uM_{i}_1 --output_folder=only_pored/AZD_test/inference_jan_2025", shell=True)
# for i in range(1, 6):
#   subprocess.run(f"python clumps_table.py --input_path=inference/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_{i}_1_Merged.tif_AZD_jan2025_250nMAZD_{i}_1.output.png --prediction_type=clumps --filter_type=otsu --save_image_as=AZD_250nM_{i}_1 --output_folder=only_pored/AZD_test/inference_jan_2025", shell=True)
#   subprocess.run(f"python clumps_table.py --input_path=inference/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_{i}_1_Merged.tif_AZD_jan2025_100nMAZD_{i}_1.output.png --prediction_type=clumps --filter_type=otsu --save_image_as=AZD_100nM_{i}_1 --output_folder=only_pored/AZD_test/inference_jan_2025", shell=True)
#   subprocess.run(f"python clumps_table.py --input_path=inference/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_{i}_1_Merged.tif_AZD_jan2025_DMSO_{i}_1.output.png --prediction_type=clumps --filter_type=otsu --save_image_as=AZD_DMSO_{i}_1 --output_folder=only_pored/AZD_test/inference_jan_2025", shell=True)

# with open('splits/pores_only_test_12-2024.csv', newline='') as csvfile:

#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

#     for row in spamreader:
#         row = [i.strip(" ") for i in row]
#         row = [i.strip(",") for i in row]
#         print(row)
#         shutil.copy(row[0], f"pore_nopore_test_folder/BASE/{os.path.basename(row[0])}")
#         shutil.copy(row[1], f"pore_nopore_test_folder/ANNOT_former/{os.path.basename(row[0])}")
#         subprocess.run(f"python clumps_table.py --input_path={row[1]} --output_folder=pore_nopore_test_folder/clumps --prediction_type=clumps --filter_type=otsu", shell=True)

# for path in glob.glob("AZD_source/ANNOT_prep/*.png"):
#     subprocess.run(f"python clumps_table.py --input_path={path} --output_folder=AZD_source/clumps --prediction_type=clumps --filter_type=confidence", shell=True)
#     pass

        # print(', '.join(row))

# Make Trio Files

# Get CSV Files

# Assemble TrioClips
# Sort into folder
# Run FOR loop

# for path in glob.glob("pore_nopore_test_folder/BASE/*.*"):
#     print(os.path.basename(path))
# for annot_path in glob.glob("pore_nopore_test_folder/clumps/*.*"):
#     print(os.path.basename(annot_path)[:-len("_ANNOT.csv")])

# csvlist = [os.path.basename(p) for p in glob.glob("pore_nopore_test_folder/clumps/*.*")]

# trio_list = [[f"pore_nopore_test_folder/BASE/{pth[:-len('_ANNOT.csv')]}.tif", f"pore_nopore_test_folder/ANNOT_former/{pth[:-len('_ANNOT.csv')]}.tif", f"pore_nopore_test_folder/clumps/{pth}",] for pth in csvlist]

# for t in trio_list:
#     for p in t:
#         assert os.path.isfile(p), f"Could not be found: {p}"


# for t in trio_list:
#     base_file, annot_file, csv_file = t[0], t[1], t[2]
#     # LOAD BASE IMAGE
#     base_img = Image.open(base_file)
#     # LOAD ANNOT IMAGE (maybe not now)

#     # with open(csv_file, 'r') as csv_reader:
#     #     header_row = next(csv_reader)
#     #     for i, row in enumerate(csv_reader):
#     #         row = row.strip().split(",")
#     #         type = row[14]
#     #         # print(type)
#     #         clump_num = int(row[2])
#     #         # print(clump_num)
#     #         # check if the folder exists
#     #         y0, x0, y1, x1 = int(row[3]), int(row[4]), int(row[5]), int(row[6])
#     #         yc, xc = (y1+y0)//2, (x1+x0)//2
#     #         yK, xK = yc-32, xc-32
#     #         crop = base_img.crop([xc-36, yc-36, xc+36, yc+36])
#     #         crop.save(f"andrew_doublecheck/{type}/{os.path.basename(base_file)[:-4]}_{clump_num:04d}.png")
#     df = pd.read_csv(csv_file)
#     for index, row in df.iterrows():
#         type = row["Notes"]
#         # print(type)
#         clump_num = int(row["label"])
#         # print(clump_num)
#         # check if the folder exists
#         y0, x0, y1, x1 = int(row["bbox-0"]), int(row["bbox-1"]), int(row["bbox-2"]), int(row["bbox-3"])
#         yc, xc = (y1+y0)//2, (x1+x0)//2
#         yK, xK = yc-32, xc-32
#         crop = base_img.crop([xc-36, yc-36, xc+36, yc+36])
#         crop.save(f"andrew_doublecheck/{type}/{os.path.basename(base_file)[:-4]}_{clump_num:04d}.png")


# Get a window of size 72 around each stomata by BBOX

for file in glob.glob("AZD_2025/start_tifs/*.tif"):
  subprocess.run(f'python inference.py --model=checkpoints/2025-02-12_19h-14m-25s/last.e029.pth --input="{file}" --overlap=128 --outputname="_AZD_INF_2025"', shell=True)
