import subprocess
import tqdm

import os
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
    i = 1
    # for filename in os.listdir("SCD_training_data/source_images/BASE/basl-2_5dpg_cotyledons"):
    #     command_infer = f"python inference.py --model=checkpoints/2024-04-22_11h-16m-11s/last.e029.pth --input=SCD_training_data/source_images/BASE/basl-2_5dpg_cotyledons/{filename}"
    #     subprocess.run(command_infer, shell=True)
    #     command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=basl-2_5dpg_COT{i:02d}"
    #     subprocess.run(command_clump, shell=True)
    #     i += 
    print("B")
    for filename in os.listdir("SCD_training_data/source_images/BASE/trm678_5dpg_cotyledons"):
        os.rename(f"SCD_training_data/source_images/BASE/trm678_5dpg_cotyledons/{filename}", f"SCD_training_data/source_images/BASE/trm678_5dpg_cotyledons/{filename.replace(' ', '-')}")
        command_infer = f"python inference.py --model=checkpoints/2024-04-22_11h-16m-11s/last.e029.pth --input=SCD_training_data/source_images/BASE/trm678_5dpg_cotyledons/{filename}"
        subprocess.run(command_infer, shell=True)
        command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=trm678_5dpg_COT{i:02d}"
        subprocess.run(command_clump, shell=True)
        i += 1
    i = 1
    print("A")
    # #[os.rename(file, file.replace(" ", "-")) for files in os.listdir("SCD_training_data\\source_images\\BASE\\basl-2_5dpg_cotyledons") for file in files]
    # for filename in os.listdir("SCD_training_data/source_images/BASE/TMMpLNG1_3dpg_cotyledons"):
    #     print(filename)
    #     os.rename(f"SCD_training_data/source_images/BASE/TMMpLNG1_3dpg_cotyledons/{filename}", f"SCD_training_data/source_images/BASE/TMMpLNG1_3dpg_cotyledons/{filename.replace(' ', '-')}")
    #     command_infer = f"python inference.py --model=checkpoints/2024-04-22_11h-16m-11s/last.e029.pth --input=SCD_training_data/source_images/BASE/TMMpLNG1_3dpg_cotyledons/{filename}"
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
    print("E")
    for filename in os.listdir("SCD_training_data/source_images/BASE/WT_8dpg_true-leaves"):
        os.rename(f"SCD_training_data/source_images/BASE/WT_8dpg_true-leaves/{filename}", f"SCD_training_data/source_images/BASE/WT_8dpg_true-leaves/{filename.replace(' ', '-')}")
        command_infer = f"python inference.py --model=checkpoints/2024-04-22_11h-16m-11s/last.e029.pth --input=SCD_training_data/source_images/BASE/WT_8dpg_true-leaves/{filename}"
        subprocess.run(command_infer, shell=True)
        command_clump = f"python clumpfinder.py --input_path=inference/{filename}.output.png --closing_threshold=80 --opening_threshold=120 --save_to=WT_8dpg_true-leaves_TRU{i:02d}"
        subprocess.run(command_clump, shell=True)
        i += 1
        # command_infer = f"python clumpfinder.py  --input_path= --output_folder=inference/tables --closing_threshold=80 --opening_threshold=120"
        # command_clump = f""
# clumpfinder_auto("area,axis_major_length")
clumps_table_auto()