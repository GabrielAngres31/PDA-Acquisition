import subprocess
import tqdm

import os
import glob

model_path = "checkpoints/2024-06-27_11h-31m-26s/last.e029.pth"
old_model_path = "checkpoints/2024-04-22_11h-16m-11s/last.e029.pth"
outline_model_path = "checkpoints/2024-07-24_00h-32m-14s/last.e029.pth"
# def clumpfinder_auto(properties):
#     for i in tqdm.tqdm(range(1, 15)):
#         command = f"python clumpfinder.py --input_path=SCD_training_data/source_images/ANNOTATION/cotE{i:02d}_STOMATA_MASKS.tiff --closing_threshold=80 --opening_threshold=120 --scatter_plot=1 --save_to=cotE{i:02d} --properties='{properties}'"
#         subprocess.run(command, shell=True)
#     for i in tqdm.tqdm(range(1, 7)):
#         command = f"python clumpfinder.py --input_path=SCD_training_data/source_images/ANNOTATION/cot{i}_STOMATA_MASKS.tiff --closing_threshold=80 --opening_threshold=120 --scatter_plot=1 --save_to=cot{i} --properties='{properties}'"
#         subprocess.run(command, shell=True)

# clumpfinder_auto("area,axis_major_length")
# def clumps_table_auto():
#     for i in tqdm.tqdm(range(1, 15)):
#         command = f"python clumps_table.py --input_path=SCD_training_data/source_images/ANNOTATION/cotE{i:02d}_STOMATA_MASKS.tiff --output_folder=inference/tables --closing_threshold=80 --opening_threshold=120"
#         subprocess.run(command, shell=True)
#     for i in tqdm.tqdm(range(1, 7)):
#         command = f"python clumps_table.py --input_path=SCD_training_data/source_images/ANNOTATION/cot{i}_STOMATA_MASKS.tiff --output_folder=inference/tables --closing_threshold=80 --opening_threshold=120"
#         subprocess.run(command, shell=True)
def clumps_table_auto():
    # files_list = [
    #     "basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged.tif.output.png",
    #     "basl-2_5_COT_03_rotated_MAX_basl-2_5dpg_110321_2_1_abaxial_merged.tif.output.png",
    #     "basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged.tif.output.png",
    #     "cot1_STOMATA_MASKS.tiff",
    #     "cot2_STOMATA_MASKS.tiff",
    #     "cot3_STOMATA_MASKS.tiff",
    #     "cot4_STOMATA_MASKS.tiff",
    #     "cot5_STOMATA_MASKS.tiff",
    #     "cot6_STOMATA_MASKS.tiff",
    #     "cotE01_STOMATA_MASKS.tiff",
    #     "cotE02_STOMATA_MASKS.tiff",
    #     "cotE03_STOMATA_MASKS.tiff",
    #     "cotE04_STOMATA_MASKS.tiff",
    #     "cotE05_STOMATA_MASKS.tiff",
    #     "cotE06_STOMATA_MASKS.tiff",
    #     "cotE07_STOMATA_MASKS.tiff",
    #     "cotE08_STOMATA_MASKS.tiff",
    #     "cotE09_STOMATA_MASKS.tiff",
    #     "cotE10_STOMATA_MASKS.tiff",
    #     "cotE11_STOMATA_MASKS.tiff",
    #     "cotE12_STOMATA_MASKS.tiff",
    #     "cotE13_STOMATA_MASKS.tiff",
    #     "cotE14_STOMATA_MASKS.tiff",
    #     "trm678_5_COT_02.tiff",
    #     "UBQ10pLNG1_3_COT_10.png"
    # ]
    # for filename in files_list:
    for filename in glob.glob("SCD_training_data\\source_images\\ANNOTATION\\OUTLINES\\annotations\\*.tiff"):
        print(filename)
        # command_table = f"python clumps_table.py\
        #                     --model={model_path} \
        #                     --input=SCD_training_data/source_images/ANNOTATION/{filename} \
        #                     --overlap=128"
        # subprocess.run(command_table, shell=True)
        command_clump = f"python clumps_table.py \
                        --output_folder=inference/clump_data/OUTLINES\
                        --input_path={filename}"
                        # --closing_threshold=10000 \
                        # --opening_threshold=0 \
        subprocess.run(command_clump, shell=True)
    # for filename in os.listdir("SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons"):
    #     print(filename)
    #     command_infer = f"python inference.py --model={model_path} --input=SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons/{filename} --overlap=128"
    #     subprocess.run(command_infer, shell=True)
    #     command_clump = f"python clumps_table.py \
    #                     --closing_threshold=80 \
    #                     --opening_threshold=120 \
    #                     --output_folder=inference/clump_data\
    #                     --input_path=inference/{filename}.output.png"
    #     subprocess.run(command_clump, shell=True)
#     for filename in os.listdir("SCD_training_data/source_images/BASE/basl-2_5dpg_cotyledons"):
#         command_infer = f"python inference.py --model={model_path} --input=SCD_training_data/source_images/BASE/basl-2_5dpg_cotyledons/{filename}"
#         subprocess.run(command_infer, shell=True)
#         command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=basl-2_5dpg_COT{i:02d}"
#         subprocess.run(command_clump, shell=True)
#         i += 1
    # print("B")
    # for filename in os.listdir("SCD_training_data/source_images/BASE/trm678_5dpg_cotyledons"):
    #     os.rename(f"SCD_training_data/source_images/BASE/trm678_5dpg_cotyledons/{filename}", f"SCD_training_data/source_images/BASE/trm678_5dpg_cotyledons/{filename.replace(' ', '-')}")
    #     command_infer = f"python inference.py --model=checkpoints/2024-04-22_11h-16m-11s/last.e029.pth --input=SCD_training_data/source_images/BASE/trm678_5dpg_cotyledons/{filename}"
    #     subprocess.run(command_infer, shell=True)
    #     command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=trm678_5dpg_COT{i:02d}"
    #     subprocess.run(command_clump, shell=True)
    #     i += 1
    # i = 1
    print("A")
    #[os.rename(file, file.replace(" ", "-")) for files in os.listdir("SCD_training_data\\source_images\\BASE\\basl-2_5dpg_cotyledons") for file in files]
    # for filename in os.listdir("SCD_training_data/source_images/BASE/TMMpLNG1_3dpg_cotyledons"):
    #     print(filename)
    #     os.rename(f"SCD_training_data/source_images/BASE/TMMpLNG1_3dpg_cotyledons/{filename}", f"SCD_training_data/source_images/BASE/TMMpLNG1_3dpg_cotyledons/{filename.replace(' ', '-')}")
    #     command_infer = f"python inference.py --model={model_path} --input=SCD_training_data/source_images/BASE/WT_8dpg_true-leaves/{filename}"
    #     subprocess.run(command_infer, shell=True)
    #     command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=TMMpLNG1_3dpg_COT{i:02d}"
    #     subprocess.run(command_clump, shell=True)
    #     i += 1
    # i = 1
    print("C")
    # for filename in os.listdir("SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons"):
    #     os.rename(f"SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons/{filename}", f"SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons/{filename.replace(' ', '-')}")
    #     command_infer = f"python inference.py --model=checkpoints/2024-04-22_11h-16m-11s/last.e029.pth --input=SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons/{filename}"
    #     subprocess.run(command_infer, shell=True)
    #     command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=UBQ10pLNG1_3dpg_COT{i:02d}"
    #     subprocess.run(command_clump, shell=True)
    #     i += 1
    # i = 1
    print("D")
    # for filename in os.listdir("SCD_training_data/source_images/BASE/WT_3dpg_cotyledons"):
    #     os.rename(f"SCD_training_data/source_images/BASE/WT_3dpg_cotyledons/{filename}", f"SCD_training_data/source_images/BASE/WT_3dpg_cotyledons/{filename.replace(' ', '-')}")
    #     command_infer = f"python inference.py --model=checkpoints/2024-04-22_11h-16m-11s/last.e029.pth --input=SCD_training_data/source_images/BASE/WT_3dpg_cotyledons/{filename}"
    #     subprocess.run(command_infer, shell=True)
    #     command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=WT_3dpg_COT{i:02d}"
    #     subprocess.run(command_clump, shell=True)
    #     i += 1
    # i = 1
    # print("E")
    # for filename in os.listdir("SCD_training_data/source_images/BASE/WT_8dpg_true-leaves"):
    #     os.rename(f"SCD_training_data/source_images/BASE/WT_8dpg_true-leaves/{filename}", f"SCD_training_data/source_images/BASE/WT_8dpg_true-leaves/{filename.replace(' ', '-')}")
    #     command_infer = f"python inference.py --model={model_path} --input=SCD_training_data/source_images/BASE/WT_8dpg_true-leaves/{filename}"
    #     subprocess.run(command_infer, shell=True)
    #     command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=WT_8dpg_true-leaves_TRU{i:02d}"
    #     subprocess.run(command_clump, shell=True)
    #     i += 1
    #     command_infer = f"python clumpfinder.py  --input_path= --output_folder=inference/tables --closing_threshold=80 --opening_threshold=120"
    #     command_clump = f""
# clumpfinder_auto("area,axis_major_length")
clumps_table_auto()
