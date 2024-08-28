import os
import subprocess

for i in range(16, 30):
    # command_infer = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e{i:03}.pth \
    #             --input="SCD_training_data/source_images/ANNOTATION/OUTLINES/pdns/UBQ10pLNG1_cot/MASK__UBQ10pLNG1_3_COT_02_MAX_LineScreen_UBQ10pLNG1_3dpg_121723.lif - MUL_L5_9_Merged.tiff" \
    #             --overlap=128 \
    #             --outputdir="./inference/consequent/" \
    #             --outputname=_{i:03}'
    # command_infer = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e{i:03}.pth \
    #             --input="SCD_training_data/source_images/BASE/WT_3dpg_cotyledons/WT_3_COT_03_rotated_MAX_BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_3dpg_101821_8_1_abaxial_merged.tif" \
    #             --overlap=128 \
    #             --outputdir="./inference/consequent/" \
    #             --outputname=_test1_{i:03}'
    # command_infer = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e{i:03}.pth \
    #             --input="SCD_training_data/source_images/BASE/cot6.tif" \
    #             --overlap=128 \
    #             --outputdir="./inference/consequent/" \
    #             --outputname=_test2_{i:03}'
    # command_assess = f'python clumps_table.py --input_path="inference/consequent/cot6.tif_test2_{i:03}.output.png" --output_folder="inference/consequent" --save_image_as="_test2_{i:03}"'
    
    command_infer_outliner = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e{i:03}.pth \
                --input="SCD_training_data/source_images/BASE/basl-2_5dpg_cotyledons/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged.tif" \
                --overlap=64 \
                --outputdir="./inference/consequent/" \
                --outputname=_test4_inf_out_{i:03}'
    command_assess_outliner = f'python clumps_table.py --input_path="inference/consequent/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged.tif_test4_inf_out_{i:03}.output.png" --output_folder="inference/consequent" --prediction_type="outlines" --save_image_as="_clmp_test4_out_{i:03}"'
    
    command_infer_blobber = f'python inference.py --model=checkpoints/2024-07-11_14h-36m-00s/last.e{i:03}.pth \
                --input="SCD_training_data/source_images/BASE/basl-2_5dpg_cotyledons/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged.tif" \
                --overlap=64 \
                --outputdir="./inference/consequent/" \
                --outputname=_test4_inf_blb_{i:03}'
    command_assess_blobber = f'python clumps_table.py --input_path="inference/consequent/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged.tif_test4_inf_blb_{i:03}.output.png" --output_folder="inference/consequent" --prediction_type="clumps" --save_image_as="_test4_clmp_blb_{i:03}"'
    # command = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e{i:03}.pth \
    #             --input="SCD_training_data/source_images/BASE/cot6.tif" \
    #             --overlap=128 \
    #             --outputdir="./inference/consequent/" \
    #             --outputname=_{i:03}'
    # command = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e{i:03}.pth \
    #             --input="SCD_training_data/source_images/BASE/8dpg/BRXL2pBRXL2-YFP_ML1pmCherry-RCI2A_8dpg_TL_121721_TL2_abaxial_merged_rotated-0002.tif" \
    #             --overlap=128 \
    #             --outputdir="./inference/consequent/" \
    #             --outputname=_{i:03}'
    print(f"Inferring Image {i} with Outline")
    subprocess.run(command_infer_outliner, shell=True)
    print(f"Assessing Image {i} with Outline")
    subprocess.run(command_assess_outliner, shell=True)
    print(f"Inferring Image {i} with Blobs")
    subprocess.run(command_infer_blobber, shell=True)
    print(f"Assessing Image {i} with Blobs")
    subprocess.run(command_assess_blobber, shell=True)

