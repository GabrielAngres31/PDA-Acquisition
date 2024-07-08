# visualizers_scrapbox

import subprocess
import tqdm

import os

def scatter_auto():
    for filename in os.listdir("inference/clump_data"):
        print(filename[:-4])
        assert filename, "SHIT BOY"
        if filename == "aggregate.csv":
            continue
        command = f'python visualizers.py \
            --source_data=inference/clump_data/cot6.csv \
            --histograms="area,axis_major_length,axis_minor_length,eccentricity" \
            --scatterplots="area,axis_major_length|area,axis_minor_length|axis_major_length,axis_minor_length" \
            --save_as={filename[:-4]}'                                                                 
        subprocess.run(command, shell=True)

scatter_auto()