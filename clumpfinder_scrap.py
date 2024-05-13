import subprocess

for i in range(1, 15):
    command = f"python clumpfinder.py --input_path=SCD_training_data/source_images/ANNOTATION/cotE{i:02d}_STOMATA_MASKS.tiff --closing_threshold=80 --opening_threshold=120 --scatter_plot=1 --save_to=cotE{i:02d}"
    subprocess.run(command, shell=True)
for i in range(1, 7):
    command = f"python clumpfinder.py --input_path=SCD_training_data/source_images/ANNOTATION/cot{i}_STOMATA_MASKS.tiff --closing_threshold=80 --opening_threshold=120 --scatter_plot=1 --save_to=cot{i}"
    subprocess.run(command, shell=True)