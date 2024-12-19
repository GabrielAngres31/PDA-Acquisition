import numpy as np
import pandas as pd

import subprocess
import os

import sys
import glob


# get splits file
# iterate over paired data
# export all crops of a clump of a SET SIZE to a given folder

from PIL import Image

splits_file = pd.read_csv("splits/pores_only_test.csv")

import site
print(site.getsitepackages())

make_tables = False


# for index, row in splits_file.iterrows():
#     # continue
#     # print(row[0])
#     # relpath = os.path.relpath(os.path.join(row[1]), "C:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\")
#     relpath = os.path.relpath(row.iloc[1].replace(" ", ""))
#     filename = os.path.basename(row.iloc[1])
#     # print(row[1])
#     # print(relpath)
#     if make_tables:
#         subprocess.run(f'c:\\Users\\Gabriel\\Documents\\GitHub\\PDA-Acquisition\\venv\\Scripts\\python.exe clumps_table.py --input_path="{row.iloc[1].replace(" ", "")}" --output_folder="SCD_training_data/mbn_training/bulks/tables_mbn_bulks" --prediction_type="clumps" --filter_type="otsu"', shell=True)
# for table in glob.glob("SCD_training_data/mbn_training/bulks/tables_mbn_bulks/*"):
    # table = "SCD_training_data/mbn_training/bulks/tables_mbn_bulks/{filename[:-4]}.csv"
    # # print(table)
    # source = pd.read_csv(f"SCD_training_data/mbn_training/bulks/tables_mbn_bulks/{filename[:-4]}.csv")
    # columns = ['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']
    # bboxes = list(zip(*[source[col] for col in columns]))

    # if not bboxes:
    #     continue
    # print("Bounds:", bboxes[0])
    # centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in bboxes]
    # print("Centers:", centers[0])
    # crop_margin = 36
    # cboxes = [(c[1]-crop_margin, c[0]-crop_margin, c[1]+crop_margin, c[0]+crop_margin) for c in centers]

    # print("crops: ", cboxes[0])
    # # img_array = Image.open(row.iloc[1].replace(" ", ""))
    # img_array = Image.open(row.iloc[0].replace(" ", ""))
    # def safe_crop(image, coords):
    #     width, height = image.size

    #     # x1 = max(0, min(coords[0], width))
    #     # y1 = max(0, min(coords[1], height))
    #     # x2 = min(width, max(coords[0] + coords[2], 0))
    #     # y2 = min(height, max(coords[1] + coords[3], 0))


    #     # if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
    #     #     return False

    #     top = (coords[0] < 0)
    #     left = (coords[1] < 0)
    #     bottom = (coords[2] > height)
    #     right = (coords[3] > width)
    #     if top or bottom or left or right:
    #         return False
    #     else:
    #         return image.crop(coords)
    # # [img_array.crop(crop).save(f"SCD_training_data/mbn_training/bulks/{os.path.basename(row.iloc[0])[:-4]}_{crop[0]}y-{crop[1]}x.png") for crop in cboxes]
    # [safe_crop(img_array, crop).save(f"SCD_training_data/mbn_training/bulks/{os.path.basename(row.iloc[0])[:-4]}_{crop[0]}y-{crop[1]}x.png") if safe_crop (img_array, crop) else False for crop in cboxes]



table = "SCD_training_data/mbn_training/bulks/tables_mbn_bulks/{filename[:-4]}.csv"
# print(table)
# source = pd.read_csv(f"SCD_training_data/mbn_training/bulks/tables_mbn_bulks/basl-2_5_COT_05_rotated_MAX_basl-2_5dpg_110321_3_1_abaxial_merged_LOCATIONS.csv")
source = pd.read_csv(f"SCD_training_data/mbn_training/bulks/tables_mbn_bulks/basl-2_5_COT_01_rotated_MAX_basl-2_5dpg_110321_3_2_abaxial_merged_LOCATIONS.csv")
columns = ['bbox-0', 'bbox-1', 'bbox-2', 'bbox-3']
bboxes = list(zip(*[source[col] for col in columns]))


print("Bounds:", bboxes[0])
centers = [((b[0]+b[2])/2, (b[1]+b[3])/2) for b in bboxes]
print("Centers:", centers[0])
crop_margin = 36
cboxes = [(c[1]-crop_margin, c[0]-crop_margin, c[1]+crop_margin, c[0]+crop_margin) for c in centers]

print("crops: ", cboxes[0])
# img_array = Image.open(row.iloc[1].replace(" ", ""))
# img_array = Image.open("SCD_training_data/source_images/BASE/basl-2_5dpg_cotyledons/basl-2_5_COT_05_rotated_MAX_basl-2_5dpg_110321_3_1_abaxial_merged.tif")
img_array = Image.open("SCD_training_data/source_images/BASE/basl-2_5dpg_cotyledons/basl-2_5_COT_01_rotated_MAX_basl-2_5dpg_110321_3_2_abaxial_merged.tif")
def safe_crop(image, coords):
    width, height = image.size

    # x1 = max(0, min(coords[0], width))
    # y1 = max(0, min(coords[1], height))
    # x2 = min(width, max(coords[0] + coords[2], 0))
    # y2 = min(height, max(coords[1] + coords[3], 0))


    # if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
    #     return False

    top = (coords[0] < 0)
    left = (coords[1] < 0)
    bottom = (coords[2] > height)
    right = (coords[3] > width)
    if top or bottom or left or right:
        return False
    else:
        return image.crop(coords)
# [img_array.crop(crop).save(f"SCD_training_data/mbn_training/bulks/{os.path.basename(row.iloc[0])[:-4]}_{crop[0]}y-{crop[1]}x.png") for crop in cboxes]
# [safe_crop(img_array, crop).save(f"SCD_training_data/mbn_training/testing_folder_NOT_VALIDATION/basl-2_5_COT_05_rotated_MAX_basl-2_5dpg_110321_3_1_abaxial_merged_LOCATIONS_{crop[0]}y-{crop[1]}x.png") if safe_crop (img_array, crop) else print("rip") for crop in cboxes]
[safe_crop(img_array, crop).save(f"SCD_training_data/mbn_training/testing_folder_NOT_VALIDATION/basl-2_5_COT_01_rotated_MAX_basl-2_5dpg_110321_3_2_abaxial_merged_LOCATIONS_{crop[0]}y-{crop[1]}x.png") if safe_crop (img_array, crop) else print("rip") for crop in cboxes]

# if index == 20:
    # sys.exit()


#print(bboxes)