# visualizers_scrapbox

import subprocess
import tqdm

import os
import pandas as pd

import subprocess
import sys

print(sys.executable)

def scatter_auto():
    for filename in os.listdir("inference/clump_data"):
        print(filename[:-4])
        assert filename, "!!Bad Filename!!"
        if filename == "aggregate.csv":
            continue
        command = f'python visualizers.py \
            --source_data=inference/clump_data/cot6.csv \
            --histograms="area,axis_major_length,axis_minor_length,eccentricity" \
            --scatterplots="area,axis_major_length|area,axis_minor_length|axis_major_length,axis_minor_length" \
            --save_as={filename[:-4]}'                                                                 
        subprocess.run(command, shell=True)

# scatter_auto()

def basl_comparator():
    
    files_WT = ["cot1_STOMATA_MASKS.csv",
                "cot2_STOMATA_MASKS.csv",
                "cot3_STOMATA_MASKS.csv",
                "cot4_STOMATA_MASKS.csv",
                "cot5_STOMATA_MASKS.csv",
                "cot6_STOMATA_MASKS.csv",
                "cotE01_STOMATA_MASKS.csv",
                "cotE02_STOMATA_MASKS.csv",
                "cotE03_STOMATA_MASKS.csv",
                "cotE04_STOMATA_MASKS.csv",
                "cotE05_STOMATA_MASKS.csv",
                "cotE06_STOMATA_MASKS.csv",
                "cotE07_STOMATA_MASKS.csv",

                "cotE08_STOMATA_MASKS.csv",
                "cotE09_STOMATA_MASKS.csv",
                "cotE10_STOMATA_MASKS.csv",
                "cotE11_STOMATA_MASKS.csv",
                "cotE12_STOMATA_MASKS.csv",
                "cotE13_STOMATA_MASKS.csv",
                "cotE14_STOMATA_MASKS.csv"
                ]
    
    files_basl = ["basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged.tif.output.csv",
                  "basl-2_5_COT_03_rotated_MAX_basl-2_5dpg_110321_2_1_abaxial_merged.tif.output.csv",
                  "basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged.tif.output.csv"]

    WT_combined = pd.DataFrame()
    basl_combined = pd.DataFrame()


    for file in files_WT:
        df = pd.read_csv(f"inference/clump_data/{file}")
        WT_combined = pd.concat([WT_combined, df])

    for file in files_basl:
        df = pd.read_csv(f"inference/clump_data/{file}")
        basl_combined = pd.concat([basl_combined, df])

    print(WT_combined.head())
    print(basl_combined.head())
    WT_combined.to_csv("inference/clump_data/aggregate_folders/WT_combined.csv")
    basl_combined.to_csv("inference/clump_data/aggregate_folders/basl_combined.csv")


    wt_command = 'python visualizers.py\
                --source_data=inference/clump_data/aggregate_folders/WT_combined.csv\
                --scatterplots="area,area_bbox|area,area_convex|area_bbox,area_convex"\
                --save_as="basl-separation-test/wt_profile"\
                --xmax=7000\
                --ymax=7000\
                    '
    subprocess.run(wt_command, shell=True)
    basl_command = 'python visualizers.py\
                --source_data=inference/clump_data/aggregate_folders/basl_combined.csv\
                --scatterplots="area,area_bbox|area,area_convex|area_bbox,area_convex"\
                --save_as="basl-separation-test/basl_profile"\
                --xmax=7000\
                --ymax=7000\
                    '
    subprocess.run(basl_command, shell=True)


# basl_comparator()

dim_list = {"axis_major_length,axis_minor_length":[100,100],
            "eccentricity,perimeter":[1,250],
            "area,area_convex":[2500,2500],
            "area,axis_major_length":[3000,100],
            "area,axis_minor_length":[3000,100]}

file_list = {#"inference/cot1_ANNOT.csv":"cot1", 
             #"inference/cot3_ANNOT.csv":"cot3", 
             #"inference/cot6_ANNOT.csv":"cot6", 
            #  "inference/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged_ANNOT.csv":"basl_2",
            #  "inference/basl-2_5_COT_03_rotated_MAX_basl-2_5dpg_110321_2_1_abaxial_merged_ANNOT.csv":"basl_3",
            #  "inference/basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged_ANNOT.csv":"basl_4",
             "inference/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged_ANNOT_modded.csv":"basl_2",
             "inference/basl-2_5_COT_03_rotated_MAX_basl-2_5dpg_110321_2_1_abaxial_merged_ANNOT_modded.csv":"basl_3",
             "inference/basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged_ANNOT_modded.csv":"basl_4",
             }

for file in file_list:
    for dim in dim_list:
        subprocess.run(f'python visualizers.py --source_data={file} --scatterplots={dim} --save_as="{file_list[file]}_glance" --xmax={dim_list[dim][0]} --ymax={dim_list[dim][1]}', shell=True)