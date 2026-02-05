# ridgeplot_scraopbox.py

import pandas as pd
import joypy
import numpy as np
import matplotlib.pyplot as plt

# df = pd.read_csv("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/section_based_training_07-2025/CLUMPS/ridgeplots_wt/full.csv")

# fig, axes = joypy.joyplot(df
#                           , column=['area']
#                         #   , overlap=2.5
#                         #   , fill = False
#                           , by="ID"
#                         #   , background='k'
#                         #   , linecolor="w"
#                         #   , hist = True
#                         #   , bins = 30
#                           )
# plt.title('Placeholder'
#         #   , fontsize=14
#         #   , color='grey'
#         #   , alpha=1)
#         )
# # plt.rc("font", size=12)
# plt.xlabel('placeholder_a', fontsize=14, color='grey', alpha=1)
# plt.ylabel('placeholder_b', fontsize=8, color='grey', alpha=1)
# plt.show()


import shutil

# adf_pth = "C:/Users/Gabriel/Downloads/samples_process_07302025/pubsplit_trn_08-2025.csv"
# vdf_pth = "C:/Users/Gabriel/Downloads/samples_process_07302025/pubsplit_val_08-2025.csv"
# adf = pd.read_csv(adf_pth, names = ["base", "annot"])
# vdf = pd.read_csv(vdf_pth, names = ["base", "annot"])

# for i in adf["annot"]:
    # print(i)
    # shutil.copy(i, "C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/trn/")

# for i in vdf["annot"]:
    # print(i)
    # shutil.copy(i, "C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/val/")

import glob
import subprocess

# for b in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/trn/*.png"):
#     subprocess.run(f'python clumps_table_SUF.py --input_path={b} --output_folder="C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_trn"/')
# for v in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/val/*.png"):
#     subprocess.run(f'python clumps_table_SUF.py --input_path={v} --output_folder="C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_val"/')

# for b in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/trn_no-edge/*.png"):
#     subprocess.run(f'python clumps_table_SUF.py --input_path={b} --output_folder="C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_trn_no-edge"/')
# for v in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/val_no-edge/*.png"):
#     subprocess.run(f'python clumps_table_SUF.py --input_path={v} --output_folder="C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_val_no-edge"/')

import os
from PIL import Image

# def onborder(image, coords):
#     width, height = image.size
#     up, left, down, right = coords[0], coords[1], coords[2], coords[3]
#     if up == 0 or left == 0 or down == height or right == width:
#         return True
#     else:
#         return False

# for b in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/val/*.png"):
#     b_img = Image.open(b).convert("L")
#     b_pth = os.path.splitext(os.path.basename(b))[0]
#     b_csv_pth = "C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_val/" + b_pth + ".csv"
#     b_df = pd.read_csv(b_csv_pth)

#     for r in b_df.itertuples():
#         cor = (r[3], r[4], r[5], r[6])
#         decision = onborder(b_img, cor)
#         if decision:
#             for j in range(cor[0], cor[2]):
#                 for i in range(cor[1], cor[3]):
#                     b_img.putpixel((i, j), 0)
#         b_img.save(f"C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/val_no-edge/{b_pth}.png")
        # print(r[3])
        # print(r)

    # rows_to_keep = b_df.apply(
    #     lambda row: not onborder(row['bbox-0'], row['bbox-1'], row['bbox-2'], row['bbox-3']),
    #     axis=1
    # )

    # Filter the DataFrame to keep only the desired rows
    # filtered_df = df[rows_to_keep]
# for v in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/val/*.png"):





# input_dict = {}

# # l = 0

# for p in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_trn/*.csv"):
#     # if l > 10: break
#     p_pth = os.path.basename(p)
#     input_dict[p] = input(f"INPUT dpg value for {p_pth}: ")
#     # l+=1
# print("----------------")
# print("----------------")
# print("----------------")
# for q in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_val/*.csv"):
#     # if l > 10: break
#     q_pth = os.path.basename(q)
#     input_dict[q] = input(f"INPUT dpg value for {q_pth}: ")

# import csv 
# with open("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/dpg_vals.csv", mode='w', newline='') as file:
#     # Create a DictWriter object
#     writer = csv.DictWriter(file, fieldnames=["file", "dpg"])

#     # Write the header row using the fieldnames
#     writer.writeheader()

#     # Write the data rows from the list of dictionaries
#     for key, value in input_dict.items():
#         writer.writerow({"file":key, "dpg": value})


# --------------------


# out_dictlist = []

# dictcsv = pd.read_csv("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/dpg_vals.csv")
# for t in dictcsv.itertuples():
#     staged_csv = pd.read_csv(t[1])
#     csv_len = len(staged_csv)
#     # print(f"{os.path.basename(t[1])} contains {csv_len} stomata at [{t[2]}] dpg")
#     out_dictlist.append({'file':t[1], "dpg":t[2], "count":csv_len, "de-edged":"FALSE"})


# fictcsv = pd.read_csv("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/dpg_vals.csv")
# for u in fictcsv.itertuples():
#     # print(u)
#     try:
#         u_pth = "C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_trn_no-edge/"+os.path.basename(u[1])
#     except:
#         u_pth = "C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_val_no-edge/"+os.path.basename(u[1])
#     try:
#         staged_csv = pd.read_csv(u_pth)
#         csv_len = len(staged_csv)
#         # print(f"{os.path.basename(u[1])} contains {csv_len} stomata at [{u[2]}] dpg")
#         out_dictlist.append({'file':u[1], "dpg":u[2], "count":csv_len, "de-edged":"TRUE"})
#     except:
#         try:
#             u_pth = "C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/clumps_val_no-edge/"+os.path.basename(u[1])
#             staged_csv = pd.read_csv(u_pth)
#             csv_len = len(staged_csv)
#             # print(f"{os.path.basename(u[1])} contains {csv_len} stomata at [{u[2]}] dpg")
#             out_dictlist.append({'file':u[1], "dpg":u[2], "count":csv_len, "de-edged":"TRUE"})
#         except:
#             print(f"COULDN'T FIND FILE: {u_pth}")

# print(out_dictlist)

# import csv

# with open('C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/count_info.csv', 'w', newline='') as csvfile_q:
#     # Create a DictWriter object
#     writer = csv.DictWriter(csvfile_q, fieldnames=["file", "dpg", "count", "de-edged"])

#     # Write the header row
#     writer.writeheader()

#     # Write the data rows
#     writer.writerows(out_dictlist)

# --------------------

counts_df = pd.read_csv("C:/Users/Gabriel/Downloads/samples_process_07302025/USED_ANNOT/count_info.csv")
# counts_df_nedge  = counts_df[counts_df["de-edged"] == True][~counts_df["file"].str.contains("basl")]
# counts_df_edge  = counts_df[counts_df["de-edged"] == False][~counts_df["file"].str.contains("basl")]


# counts_df_nedge_wholes = counts_df[counts_df["de-edged"] == True][counts_df["file"].str.contains("|".join(['cot1_ANNOT', 'cot2_ANNOT', 'cot3_ANNOT', 'cot4_ANNOT', 'cot5_ANNOT', 'cotE01_ANNOT','cotE02_ANNOT','cotE06_ANNOT','cotE07_ANNOT','cotE08_ANNOT','cotE09_ANNOT','cotE10_ANNOT']))]
# counts_df_nedge_segmnt = counts_df[counts_df["de-edged"] == True][~counts_df["file"].str.contains("|".join(['cot1_ANNOT', 'cot2_ANNOT', 'cot3_ANNOT', 'cot4_ANNOT', 'cot5_ANNOT', 'cotE01_ANNOT','cotE02_ANNOT','cotE06_ANNOT','cotE07_ANNOT','cotE08_ANNOT','cotE09_ANNOT','cotE10_ANNOT']))]

# print(counts_df_edge)

# print(counts_df_nedge_wholes.groupby('dpg')['count'].sum())
# print(counts_df_nedge_segmnt.groupby('dpg')['count'].sum())

counts_df_degded_grouped = counts_df[counts_df["de-edged"] == True].groupby(["dpg", "type", "frmt", "set"])["count"].sum()
print(counts_df_degded_grouped)

# print(counts_df_edge.groupby('dpg')['count'].sum())
# print(counts_df_nedge.groupby('dpg')['count'].sum())
