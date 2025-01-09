# visualizers_scrapbox

import subprocess
import tqdm

import os
import pandas as pd

import subprocess
import sys

import glob

import matplotlib.pyplot as plt
import joypy

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
            # #  "inference/basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged_ANNOT.csv":"basl_4",
             "inference/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged_ANNOT_modded.csv":"basl_2",
             "inference/basl-2_5_COT_03_rotated_MAX_basl-2_5dpg_110321_2_1_abaxial_merged_ANNOT_modded.csv":"basl_3",
             "inference/basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged_ANNOT_modded.csv":"basl_4",

             }

for file in file_list:
    for dim in dim_list:
        pass

# AZD_files = dict([(f,f[74:-11]) for f in glob.glob("only_pored/AZD_test/inference_jan_2025/*.csv")])

AZD_files = {
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_1_1_Merged.tif_AZD_jan2025_100nMAZD_1_1.output.csv': '_AZD_jan2025_100nMAZD_1_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_2_1_Merged.tif_AZD_jan2025_100nMAZD_2_1.output.csv': '_AZD_jan2025_100nMAZD_2_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_3_1_Merged.tif_AZD_jan2025_100nMAZD_3_1.output.csv': '_AZD_jan2025_100nMAZD_3_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_4_1_Merged.tif_AZD_jan2025_100nMAZD_4_1.output.csv': '_AZD_jan2025_100nMAZD_4_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_100nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--100nMAZD_5_1_Merged.tif_AZD_jan2025_100nMAZD_5_1.output.csv': '_AZD_jan2025_100nMAZD_5_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_1_1_Merged.tif_AZD_jan2025_1uMAZD_1_1.output.csv': '_AZD_jan2025_1uMAZD_1_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_2_1_Merged.tif_AZD_jan2025_1uMAZD_2_1.output.csv': '_AZD_jan2025_1uMAZD_2_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_3_1_Merged.tif_AZD_jan2025_1uMAZD_3_1.output.csv': '_AZD_jan2025_1uMAZD_3_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_4_1_Merged.tif_AZD_jan2025_1uMAZD_4_1.output.csv': '_AZD_jan2025_1uMAZD_4_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_5_1_Merged.tif_AZD_jan2025_1uMAZD_5_1.output.csv': '_AZD_jan2025_1uMAZD_5_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_6_1_Merged.tif_AZD_jan2025_1uMAZD_6_1.output.csv': '_AZD_jan2025_1uMAZD_6_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_7_1_Merged.tif_AZD_jan2025_1uMAZD_7_1.output.csv': '_AZD_jan2025_1uMAZD_7_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_1uMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--1uMAZD_8_1_Merged.tif_AZD_jan2025_1uMAZD_8_1.output.csv': '_AZD_jan2025_1uMAZD_8_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_1_1_Merged.tif_AZD_jan2025_250nMAZD_1_1.output.csv': '_AZD_jan2025_250nMAZD_1_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_2_1_Merged.tif_AZD_jan2025_250nMAZD_2_1.output.csv': '_AZD_jan2025_250nMAZD_2_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_3_1_Merged.tif_AZD_jan2025_250nMAZD_3_1.output.csv': '_AZD_jan2025_250nMAZD_3_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_4_1_Merged.tif_AZD_jan2025_250nMAZD_4_1.output.csv': '_AZD_jan2025_250nMAZD_4_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_250nMAZD_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--250nMAZD_5_1_Merged.tif_AZD_jan2025_250nMAZD_5_1.output.csv': '_AZD_jan2025_250nMAZD_5_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_1_1_Merged.tif_AZD_jan2025_DMSO_1_1.output.csv': '_AZD_jan2025_DMSO_1_1',
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_2_1_Merged.tif_AZD_jan2025_DMSO_2_1.output.csv': '_AZD_jan2025_DMSO_2_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_3_1_Merged.tif_AZD_jan2025_DMSO_3_1.output.csv': '_AZD_jan2025_DMSO_3_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_4_1_Merged.tif_AZD_jan2025_DMSO_4_1.output.csv': '_AZD_jan2025_DMSO_4_1', 
    'only_pored/AZD_test/inference_jan_2025/MAX_DMSO_ADM100_ML1pmCherry-RCI2A_5dpg_102324.lif--DMSO_5_1_Merged.tif_AZD_jan2025_DMSO_5_1.output.csv': '_AZD_jan2025_DMSO_5_1'
}

#print(AZD_files)

# i = 0
# for file in AZD_files:
#     #print(file)
#     # if i:
#     #     break
#     # i += 1
#     for dim in dim_list:
#         # break
#         subprocess.run(f'python visualizers.py --source_data={file} --density_heatmaps={dim} --save_as="{AZD_files[file]}_AZD_graph_dh" --xmax={dim_list[dim][0]} --ymax={dim_list[dim][1]}', shell=True)
        
source_data = "only_pored/AZD_test/concatenated.csv"

def main() -> bool:
    # if not args.xmax:

    df = pd.read_csv(source_data)
    #print(args.source_data)
    #print(df)


    #print(df.head())
    assert "ID" in df, "There's only one group in this DF!"
    plots = ["area","axis_major_length","axis_minor_length","eccentricity"]
    dpg_mapping = {1:"#fde725FF", 
                2:"#90d743FF", 
                3:"#35b779FF", 
                4:"#21918cFF", 
                5:"#31688eFF", 
                6:"#443983FF", 
                7:"#440154FF"}
    AZD_mapping = {
        "100nMAZD":"#fde725FF",
        "250nMAZD":"#35b779FF",
        "1uMAZD":"#31688eFF",
        "DMSO":"#440154FF"
    }

    df = df.sort_values('Group', ascending = True)
    ranges={"area":[0,3000], "axis_major_length":[0,80], "axis_minor_length":[0,55], "eccentricity":[0,1]}
    for i, plot in enumerate(plots):
        
        fig, ax = joypy.joyplot(df.groupby("Full_ID", sort=False), by = "Group", column = plot, fade = True, x_range=ranges[plot])
        plt.title(f"Ridgeplot of {plot}")
        plt.savefig(f"reference_figures/visualizers_test/jan2025_a_{plot}_dpg_ridgeplot.png", bbox_inches = "tight")
        plt.clf()

    sorted_count_by_id = df.groupby("ID").size().sort_values(ascending=True)
    sortcount_to_dict = dict(zip([k for k in sorted_count_by_id.keys()], 
                                    [v for v in sorted_count_by_id.values]))
    
# main()

AZD_csv_path_list = [
    "only_pored/AZD_test/concat_DMSO.csv",
    "only_pored/AZD_test/concat_100nM_AZD.csv",
    "only_pored/AZD_test/concat_250nM_AZD.csv",
    "only_pored/AZD_test/concat_1uM_AZD.csv",
                     ]

tags = ["DMSO", "100nM_AZD", "250nM_AZD", "1uM_AZD"]

for i, AZD_csv_path in enumerate(AZD_csv_path_list):
    AZD_f = pd.read_csv(AZD_csv_path)
    for dim in dim_list:
        # break
        subprocess.run(f'python visualizers.py --source_data={AZD_csv_path} --scatterplots={dim} --save_as="_{tags[i]}_combined_scatterplot" --xmax={dim_list[dim][0]} --ymax={dim_list[dim][1]}', shell=True)
