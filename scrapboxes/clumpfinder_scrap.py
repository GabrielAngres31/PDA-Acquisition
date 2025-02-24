import subprocess
import tqdm

import os
import glob

model_path = "checkpoints/2024-06-27_11h-31m-26s/last.e029.pth"
old_model_path = "checkpoints/2024-04-22_11h-16m-11s/last.e029.pth"
outline_model_path = "checkpoints/2024-07-24_00h-32m-14s/last.e029.pth"

newest_model_blb = "checkpoints_useful/blobs_test_only/2024-08-30_10h-12m-05s/last.e022.pth"
newest_model_out = "checkpoints_useful/outline_test_only/2024-08-30_14h-43m-56s/last.e022.pth"


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


    # folders = [file_set[0].split("\\")[-1:][0] for file_set in os.walk("SCD_training_data\\source_images\\Akankshas_files\\")][1:]
    # # print(folders)
    # for folder in folders:
    #     print(f"inference\\Akanksha\\{folder}\\*.tif")
    #     for filename in tqdm.tqdm(glob.glob(f"SCD_training_data\\source_images\\Akankshas_files\\{folder}\\*.tif")):
    #         # command_infer = f"python inference.py --model={outline_model_path}\
    #         #     --input={filename}\
    #         #     --overlap=128\
    #         #     --outputdir=inference/Akanksha/{folder}/"
    #         # subprocess.run(command_infer, shell=True)

    #         command_clump = f"python clumps_table.py \
    #                         --output_folder=inference/tables/Akanksha_tables/{folder}\
    #                         --input_path={filename}"
    #                         # --closing_threshold=10000 \
    #                         # --opening_threshold=0 \
    #         subprocess.run(command_clump, shell=True)
                                  
        # for filename in tqdm.tqdm(glob.glob(f"inference/Akanksha/{folder}")):
        #     print(filename)
        #     command_table = f"python clumps_table.py\
        #                     --model={model_path} \
        #                     --input={filename} \
        #                     --overlap=128"
        #     subprocess.run(command_table, shell=True)
        # command_table = f"python clumps_table.py\
        #                     --model={model_path} \
        #                     --input=SCD_training_data/source_images/ANNOTATION/{filename} \
        #                     --overlap=128"
        # subprocess.run(command_table, shell=True)
        # command_clump = f"python clumps_table.py \
        #                 --output_folder=inference/clump_data/OUTLINES\
        #                 --input_path={filename}"
        #                 # --closing_threshold=10000 \
        #                 # --opening_threshold=0 \
        # subprocess.run(command_clump, shell=True)

    # RECENT
    # for filename in glob.glob("SCD_training_data\\source_images\\ANNOTATION\\OUTLINES\\annotations\\*.tiff"):
    #     print(filename)
    #     # command_table = f"python clumps_table.py\
    #     #                     --model={model_path} \
    #     #                     --input=SCD_training_data/source_images/ANNOTATION/{filename} \
    #     #                     --overlap=128"
    #     # subprocess.run(command_table, shell=True)
    #     command_clump = f"python clumps_table.py \
    #                     --output_folder=inference/clump_data/OUTLINES\
    #                     --input_path={filename}"
    #                     # --closing_threshold=10000 \
    #                     # --opening_threshold=0 \
    #     subprocess.run(command_clump, shell=True)


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
    #print("A")
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
    #print("C")
    # for filename in os.listdir("SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons"):
    #     os.rename(f"SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons/{filename}", f"SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons/{filename.replace(' ', '-')}")
    #     command_infer = f"python inference.py --model=checkpoints/2024-04-22_11h-16m-11s/last.e029.pth --input=SCD_training_data/source_images/BASE/UBQ10pLNG1_3dpg_cotyledons/{filename}"
    #     subprocess.run(command_infer, shell=True)
    #     command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=UBQ10pLNG1_3dpg_COT{i:02d}"
    #     subprocess.run(command_clump, shell=True)
    #     i += 1
    # i = 1
    #print("D")
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

    # for i, file in enumerate(glob.glob("inference/AZD_2025_inferences/*.png")):
    #     subprocess.run(f'python clean_image.py --input_path="{file}" --filter_type="confidence" --save_image_as="CLEAN_CON_{os.path.basename(file)}_"', shell=True)
    #     if i > 5:
    #         break
    for i, file in enumerate(glob.glob("cleaned_images_default/*.png")):
        subprocess.run(f'python clumps_table.py --input_path="{file}" --output_folder="cleaned_images_default/tables"', shell=True)
    #     if i > 5:
    #         break
    # for i, file in 
    # subprocess.run(f'python clumps_table.py --input_path={file} --save_image_as="CLEAN_{file}_"', shell=True)


    


    # # GET ALL THE FILES AS A LIST
    # i=0
    # for file in [os.path.basename(filename) for filename in os.listdir("evaluation_set/Total_B1-B3_wt_2024") if filename[-4:] == ".tif"]:
    #     #print(file)

    #     for j in [22]:
    #         command_infer_blb = f'python inference.py --model="checkpoints_useful/blobs_test_only/2024-08-30_10h-12m-05s/last.e{j:03d}.pth" --input="evaluation_set/Total_B1-B3_wt_2024/{file}" --overlap=64 --outputdir=./inference/match/clump/ --outputname=_e{j:03d}'
    #         subprocess.run(command_infer_blb, shell=True)
    #         command_clump_blb = f'python clumps_table.py --input_path="inference/match/clump/{file}_e{j:03d}.output.png" --output_folder="inference/match" --prediction_type="clumps" --save_image_as="match_filtered/match_blb_{i:02d}_e{j:03d}_{file}"'
    #         #f'python clumpfinder.py --input_path="inference/{file}.output.png" --closing_threshold=80 --opening_threshold=120 --save_to=match/match_blb_{i:02d}_{file}'
    #         subprocess.run(command_clump_blb, shell=True)
    #     for k in [20]:
    #         command_infer_out = f'python inference.py --model="checkpoints_useful/outline_test_only/2024-08-30_14h-43m-56s/last.e{k:03d}.pth" --input="evaluation_set/Total_B1-B3_wt_2024/{file}" --overlap=64 --outputdir=./inference/match/outline/ --outputname=_e{k:03d}'
    #         subprocess.run(command_infer_out, shell=True)
    #         #command_clump_out = f'python clumpfinder.py --input_path="inference/{file}.output.png" --closing_threshold=80 --opening_threshold=120 --save_to=match/match_out_{i:02d}_{file}'
    #         command_clump_out = f'python clumps_table.py --input_path="inference/match/outline/{file}_e{k:03d}.output.png" --output_folder="inference/match" --prediction_type="outlines" --save_image_as="match_filtered/match_out_{i:02d}_e{k:03d}_{file}"'
    #         subprocess.run(command_clump_out, shell=True)
    #         i+=1




    # l = 0
    # for file in [os.path.basename(filename) for filename in os.listdir("reference_figures/match_filtered")]:
    #     command_clump_out = f'python clumps_table.py --input_path="reference_figures/match_filtered/{file}" --output_folder="inference/match" --prediction_type="outlines"'
    #     subprocess.run(command_clump_out, shell=True)
    #     l+=1

    # START FOR LOOP

    # ESTABLISH COMMANDS FOR INFERENCE

clumps_table_auto()

# for filename in glob.glob('SCD_training_data\\source_images\\Akankshas_files\\**\\*.tif'):
#     new_name = '_'.join(filename.split(' '))  # another method 
#     os.rename(filename, new_name)
