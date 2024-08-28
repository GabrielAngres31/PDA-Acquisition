import os
import subprocess

for i in range(1, 13):
    command_infer_outliner = f'python inference.py --model=checkpoints/2024-08-05_13h-16m-30s/last.e029.pth \
                --input="evaluation_set/MAX_GJA_WT_B1_3dpg_08262024.lif - B1-3dpg-{i:02}.tif" \
                --overlap=64 \
                --outputdir="./inference/consequent/evaluator_files/" \
                --outputname=_eval_test_{i:02}_inf_OUTLINE'
    command_assess_outliner = f'python clumps_table.py --input_path="inference/consequent/evaluator_files/MAX_GJA_WT_B1_3dpg_08262024.lif - B1-3dpg-{i:02}.tif_eval_test_{i:02}_inf_OUTLINE.output.png" --output_folder="inference/consequent" --prediction_type="outlines" --save_image_as="_clmp_OUTLINE_{i:02}"'
    
    command_infer_blobber = f'python inference.py --model=checkpoints/2024-07-11_14h-36m-00s/last.e029.pth \
                --input="evaluation_set/MAX_GJA_WT_B1_3dpg_08262024.lif - B1-3dpg-{i:02}.tif" \
                --overlap=64 \
                --outputdir="./inference/consequent/evaluator_files/" \
                --outputname=_eval_test_{i:02}_inf_BLOBBY'
    command_assess_blobber = f'python clumps_table.py --input_path="inference/consequent/evaluator_files/MAX_GJA_WT_B1_3dpg_08262024.lif - B1-3dpg-{i:02}.tif_eval_test_{i:02}_inf_BLOBBY.output.png" --output_folder="inference/consequent" --prediction_type="outlines" --save_image_as="_clmp_BLOBBY_{i:02}"'

    print(f"Inferring Image {i} with Outline")
    subprocess.run(command_infer_outliner, shell=True)
    print(f"Assessing Image {i} with Outline")
    subprocess.run(command_assess_outliner, shell=True)
    print(f"Inferring Image {i} with Blobs")
    subprocess.run(command_infer_blobber, shell=True)
    print(f"Assessing Image {i} with Blobs")
    subprocess.run(command_assess_blobber, shell=True)


