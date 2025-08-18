# fast_image_subset

import cv2
import numpy as np
import glob
import os
import csv
import shutil
import tqdm
# needle_path = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/TRAINING_source/remove_ALL_smalls/BASE_smalls/3dpg_001.png"
# haystk_path = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/source_images/Uncompleted/WT_7_COT_01_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif"


# for needle_path in glob.glob():
# for i in range(18):
#     needle_path = f"C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/TRAINING_source/remove_ALL_smalls/BASE_smalls/3dpg_{(i+1):03d}.png"
#     for haystk_path in glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/source_images/Uncompleted/*.tif"):
#         needle=cv2.imread(needle_path, cv2.IMREAD_GRAYSCALE)
#         haystk=cv2.imread(haystk_path, cv2.IMREAD_GRAYSCALE)
#         # print(needle[0,0])
#         # print(needle.shape)
#         nerows, necols = needle.shape
#         haystk_searchstart = haystk[:, :] == needle[nerows//2, necols//2]
#         print(haystk_searchstart)

#         rows, cols = haystk.shape
        

#         for row in range(rows):
#             for col in range(cols):
#                 if haystk_searchstart[row, col]:
#                     z=0
#                     for i, j in [(1,0), (0,1), (-1,0), (0,-1)]:
#                         if z: break
#                         if needle[nerows//2+i, necols//2+j] != haystk[row+i, col+j]: z = 1
#                     if not z:
#                         print(f"potential match for {os.path.basename(needle_path)} found @ ({row}, {col})")



strangefiles = ["240x_1787y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_120121_3_1_abaxial_merged_rotated-0002.tif",
                "266x_1393y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
                "391x_1733y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
                "399x_1318y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
                "405x_1933y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
                "511x_1911y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
                "552x_1384y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
                "565x_817y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif",
                "737x_922y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
                "1030x_1527y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
                "1033x_109y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
                "1150x_842y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
                "1157x_882y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
                "1303x_2191y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
                "1308x_2081y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
                "1486x_1750y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
                "1496x_1132y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
                "1635x_2116y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
                "1809x_1489y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
                "1862x_1164y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
                "2069x_920y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
                "2323x_1371y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
                "2623x_1179y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
                "2751x_1556y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif"]

strangestack = ["C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/source_images/MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_120121_3_1_abaxial_merged_rotated-0002.tif",
                "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/source_images/MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_120121_4_1_abaxial_merged_rotated-0002.tif",
                "C:/Users/Gabriel/Downloads/samples_process_07302025/FORMER_COMPARE/cotE14_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
                "C:/Users/Gabriel/Downloads/samples_process_07302025/FORMER_COMPARE/cotE10_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_2_1_abaxial_merged_cropped-for-outline.tif",
                "C:/Users/Gabriel/Downloads/samples_process_07302025/FORMER_COMPARE/cotE11_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_7_1_abaxial_merged_cropped-for-outline.tif",
                "C:/Users/Gabriel/Downloads/samples_process_07302025/FORMER_COMPARE/cotE12_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
                "C:/Users/Gabriel/Downloads/samples_process_07302025/FORMER_COMPARE/cotE13_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
                "C:/Users/Gabriel/Downloads/samples_process_07302025/FORMER_COMPARE/cotE03_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
                "C:/Users/Gabriel/Downloads/samples_process_07302025/FORMER_COMPARE/cotE07_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
                "C:/Users/Gabriel/Downloads/samples_process_07302025/FORMER_COMPARE/cotE04_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif"

                ]

# for i in range(18):
#     needle_path = f"C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/TRAINING_source/remove_ALL_smalls/BASE_smalls/3dpg_{(i+1):03d}.png"
# # for needle_path in strangefiles:
#     # needle_path = f"C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/TRAINING_source/remove_ALL_smalls/BASE_smalls/7dpg_{(i+1):02d}.png"
#     # for haystk_path in glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/only_pored/source_images/Uncompleted/*.tif"):
#     fl = 1
#     for haystk_path in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/3dpg/*.tif"):
    # needle_path = f"C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/TRAINING_source/remove_ALL_smalls/BASE_smalls/7dpg_{(i+1):02d}.png"
    
    # for haystk_path in strangestack:
# newcheck = ["1030x_1527y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
# "1033x_109y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "1150x_842y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
# "1157x_882y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "1169x_1139y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_1_1_abaxial_merged.tif",
# "1187x_1794y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif",
# "1303x_2191y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
# "1308x_2081y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "1425x_825y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_3_1_abaxial_merged.tif",
# "1430x_378y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif",
# "1486x_1750y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "1496x_1132y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "1501x_1166y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif",
# "1635x_2116y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "1642x_456y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_1_1_abaxial_merged.tif",
# "1809x_1489y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "1824x_1144y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_3_1_abaxial_merged.tif",
# "1862x_1164y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
# "2069x_920y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "2323x_1371y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "2623x_1179y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "266x_1393y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "2751x_1556y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
# "391x_1733y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
# "399x_1318y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "405x_1933y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "511x_1911y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "552x_1384y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
# "565x_817y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif",
# "640x_369y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_1_1_abaxial_merged.tif",
# "737x_922y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "819x_2162y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_3_1_abaxial_merged.tif",
# "WT_3_COT_11_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_2_1_abaxial_merged_USED.tif",
# "WT_3_COT_01_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_7_1_abaxial_merged_USED.tif"]


# ["1030x_1527y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
# "1033x_109y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "1150x_842y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
# "1157x_882y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "1169x_1139y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_1_1_abaxial_merged.tif",
# "1187x_1794y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif",
# "1303x_2191y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
# "1308x_2081y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "1425x_825y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_3_1_abaxial_merged.tif",
# "1430x_378y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif",
# "1486x_1750y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "1496x_1132y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "1501x_1166y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif",
# "1635x_2116y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "1642x_456y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_1_1_abaxial_merged.tif",
# "1809x_1489y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "1824x_1144y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_3_1_abaxial_merged.tif",
# "1862x_1164y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
# "2069x_920y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "2323x_1371y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "2623x_1179y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "266x_1393y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "2751x_1556y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
# "391x_1733y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif",
# "399x_1318y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "405x_1933y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif",
# "511x_1911y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif",
# "552x_1384y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif",
# "565x_817y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif",
# "640x_369y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_1_1_abaxial_merged.tif",
# "737x_922y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif",
# "819x_2162y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_3_1_abaxial_merged.tif"]

# "RCI2A_5dpg_101321_3_1_abaxial_merged"
# "RCI2A_4dpg_101221_1_1_abaxial_merged"
# "RCI2A_5dpg_101321_7_1_adaxial_merged"
# "RCI2A_4dpg_120721"
# "RCI2A_4dpg_101921_1_1_abaxial_merged"
# "RCI2A_4dpg_101221_2_1_abaxial_merged"
# "RCI2A_4dpg_101921_3_1_abaxial_merged"
# "RCI2A_5dpg_101321_5_1_adaxial_merged"
# "RCI2A_3dpg_101821_2_1_abaxial_merged"
# "RCI2A_3dpg_101821_7_1_abaxial_merged"


with open("C:/Users/Gabriel/Downloads/samples_process_07302025/pubsplit_trn_08-2025.csv", 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        
        needle_path = row[0]
        
        fl = False
        # for haystk_path in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/sort_ALLS/*.tif"):
        # for haystk_path in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/sort_REMAINDER/*.tif"):
        for usestk_path in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/sort_USED/*.tif"):

            search_img   = cv2.imread(needle_path, cv2.IMREAD_GRAYSCALE)
            template_img = cv2.imread(usestk_path, cv2.IMREAD_GRAYSCALE)
            w, h = template_img.shape[1], template_img.shape[0]
            try:
                result = cv2.matchTemplate(search_img, template_img, cv2.TM_CCOEFF_NORMED)
            except:
                # print(f"Image might be too large: {needle_path}")
                continue
            threshold = 0.8

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val >= threshold:
                # print(f"USED: Detected for {os.path.basename(needle_path)} within {os.path.basename(usestk_path)}")
                fl = True
                if os.path.basename(needle_path) == os.path.basename(usestk_path): print(f"Whole File Detected: {os.path.basename(needle_path)}")
                else: print(f"Detected for {os.path.basename(needle_path)} within {os.path.basename(usestk_path)}")
            if fl: break
        if fl: continue
            
        for haystk_path in glob.glob("C:/Users/Gabriel/Downloads/samples_process_07302025/sort_REMAINDER/*.tif"):

            # search_img = cv2.imread(needle_path, cv2.IMREAD_GRAYSCALE)
            template_img = cv2.imread(haystk_path, cv2.IMREAD_GRAYSCALE)
            w, h = template_img.shape[1], template_img.shape[0]
            try:
                result = cv2.matchTemplate(search_img, template_img, cv2.TM_CCOEFF_NORMED)
            except:
                # print(f"Image might be too large: {needle_path}")
                continue
            threshold = 0.8

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val >= threshold:
                shutil.move(haystk_path, "C:/Users/Gabriel/Downloads/samples_process_07302025/sort_USED/")
                print(f"Detected for {os.path.basename(needle_path)} within {os.path.basename(haystk_path)}")
                break
        print(f"No Match Found for {os.path.basename(needle_path)}")

        


# No Match Found for 1030x_1527y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif
# No Match Found for 1033x_109y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif

# No Match Found for 1136x_1094y__WT_3_COT_02_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_7_2_abaxial_merged.tif
# No Match Found for 1021x_1617y__WT_3_COT_05_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101121_5_1_adaxial_merged.tif
# No Match Found for 108x_1303y__WT_3_COT_06_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101121_6_1_adaxial_merged.tif
# No Match Found for 1388x_1431y__WT_3_COT_07_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101121_6_2_adaxial_merged.tif
# No Match Found for 12x_321y__WT_3_COT_08_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101121_7_1_adaxial_merged.tif
# No Match Found for 1264x_1175y__WT_3_COT_10_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_1_2_abaxial_merged.tif
# No Match Found for 1474x_1804y__WT_3_COT_13_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_4_2_adaxial_merged.tif
# No Match Found for 1245x_1090y__WT_3_COT_14_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_5_1_adaxial_merged.tif
# No Match Found for 153x_1125y__WT_3_COT_18_112921_cot3_max_rotated_c2.tif
# No Match Found for 1655x_917y__WT_3_COT_19_112921_cot4_max_rotated_c2.tif
# No Match Found for 112x_1395y__WT_3_COT_20_112921_cot5_max_rotated_c2.tif
# No Match Found for 1060x_903y__WT_3_COT_22_112921_cot7_max_rotated_c2.tif

# No Match Found for 1252x_1096y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_120121_3_1_abaxial_merged_rotated-0002.tif
# No Match Found for 1150x_842y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif
# No Match Found for 1157x_882y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif
# No Match Found for 1169x_1139y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_1_1_abaxial_merged.tif
# No Match Found for 1187x_1794y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif
# No Match Found for 1425x_825y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_3_1_abaxial_merged.tif
# No Match Found for 1496x_1132y__rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif
# No Match Found for 1595x_1159y__MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_120121_4_1_abaxial_merged_rotated-0002.tif
# No Match Found for basl-2_5_COT_03_rotated_MAX_basl-2_5dpg_110321_2_1_abaxial_merged.tif
# No Match Found for basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged.tif
# No Match Found for cot1.tif
# No Match Found for cot2.tif
# No Match Found for cot3.tif
# No Match Found for cot4.tif
# No Match Found for cot5.tif



#  WT_3_COT_01_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_7_1_abaxial_merged_USED.tif
#  WT_3_COT_02_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_7_2_abaxial_merged_USED.tif
#  WT_3_COT_05_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101121_5_1_adaxial_merged_USED.tif
#  WT_3_COT_06_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101121_6_1_adaxial_merged_USED.tif
#  WT_3_COT_07_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101121_6_2_adaxial_merged_USED.tif
#  WT_3_COT_08_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101121_7_1_adaxial_merged_USED.tif
#  WT_3_COT_10_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_1_2_abaxial_merged_USED.tif
#  WT_3_COT_11_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_2_1_abaxial_merged_USED.tif
#  WT_3_COT_13_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_4_2_adaxial_merged_USED.tif
#  WT_3_COT_14_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_5_1_adaxial_merged_USED.tif
#  WT_3_COT_16_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_6_1_adaxial_merged.tif
#  WT_3_COT_18_112921_cot3_max_rotated_c2_USED.tif
#  WT_3_COT_20_112921_cot5_max_rotated_c2_USED.tif
#  WT_3_COT_21_112921_cot6_max_rotated_c2_USED.tif
#  WT_3_COT_22_112921_cot7_max_rotated_c2_USED.tif
#  WT_3_COT_30_112921_cot2_max_rotated_c2.tif
#  cotE02_112921_cot4_max_rotated_c2.tif
#  cotE03_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_1_1_abaxial_merged.tif
#  cotE04_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101221_2_1_abaxial_merged.tif
#  cotE06_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_3_1_abaxial_merged.tif
#  cotE05_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_101921_1_1_abaxial_merged.tif
#  cotE07_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_4dpg_120721-0002.tif
#  cotE08_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_120121_3_1_abaxial_merged_rotated-0002.tif
#  cotE09_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_120121_4_1_abaxial_merged_rotated-0002.tif
#  cotE12_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_3_1_abaxial_merged.tif
#  cotE13_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_5_1_adaxial_merged.tif
#  cotE14_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_5dpg_101321_7_1_adaxial_merged.tif
#  WT_7_COT_01_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif
#  WT_7_COT_02_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif
#  WT_7_COT_03_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif
#  WT_7_COT_04_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif
#  WT_7_COT_05_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif
#  WT_7_COT_06_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif
#  WT_7_COT_07_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif
#  WT_7_COT_08_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif
#  WT_7_COT_09_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif
#  WT_7_COT_10_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_7dpg_102221-0002.tif