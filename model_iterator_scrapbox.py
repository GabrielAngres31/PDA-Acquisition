import os
import subprocess

for i in range(0, 30):
    # command = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e{i:03}.pth \
    #             --input="SCD_training_data/source_images/ANNOTATION/OUTLINES/pdns/UBQ10pLNG1_cot/MASK__UBQ10pLNG1_3_COT_02_MAX_LineScreen_UBQ10pLNG1_3dpg_121723.lif - MUL_L5_9_Merged.tiff" \
    #             --overlap=128 \
    #             --outputdir="./inference/consequent/" \
    #             --outputname=_{i:03}'
    # command = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e{i:03}.pth \
    #             --input="SCD_training_data/source_images/BASE/cot6.tif" \
    #             --overlap=128 \
    #             --outputdir="./inference/consequent/" \
    #             --outputname=_{i:03}'
    command = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e{i:03}.pth \
                --input="SCD_training_data/source_images/BASE/8dpg/BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_8dpg_TL_121721_TL2_abaxial_merged_rotated-0002.tif" \
                --overlap=128 \
                --outputdir="./inference/consequent/" \
                --outputname=_{i:03}'
    subprocess.run(command, shell=True)