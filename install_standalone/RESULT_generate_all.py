# RESULT_generate_all.py

import numpy as np
import subprocess
from PIL import Image

from itertools import product

import glob
import typing
import os

model_of_choice = "checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth"
command =  'python inference_SUF.py --model=checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth --input="remove_ALL_smalls/BASE_smalls/cotE10.tif" --overlap=232 --outputname="o232_E10" --progress=True'

# def infer_bulk(model, files, output):
#     for file in files:
#         subprocess.run(f"python inference_SUF.py --model={model} --input={file} --overlap=127 --outputdir=inference --outputname={output}  --progress='T'", shell=True)

# def contingencies():
#     pass

def quickerator(dir:str, o:int, out:str, model:str = "checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth") -> None:
    for file in glob.glob(dir+"/*.tif"):
        name = os.path.basename(file)
        subprocess.run(f'python inference_SUF.py --model={model} --input="{file}" --overlap={o} --outputdir="{out}" --outputname="_o{o}" --progress=True --skip_empty=True', shell=True)

# quickerator("../publication_compare/base_files/AZD",                232,"../publication_compare/inference/inf_AZD/raw_inf_232/")
# quickerator("../publication_compare/base_files/basl-2_5dpg_COT/",   216,"../publication_compare/inference/inf_basl-2_5dpg_COT/")
# quickerator("../publication_compare/base_files/TMMpLNG1_3dpg_COT",  216,"../publication_compare/inference/inf_TMMpLNG1_3dpg_COT/")
# quickerator("../publication_compare/base_files/trm678_5dpg_COT",    216,"../publication_compare/inference/inf_trm678_5dpg_COT/")
# quickerator("../publication_compare/base_files/UBQ10pLNG1_3dpg_COT",216,"../publication_compare/inference/inf_UBQ10pLNG1_3dpg_COT/")
# quickerator("../publication_compare/base_files/UBQ10pOFP2",         216,"../publication_compare/inference/inf_UBQ10pOFP2/")
# quickerator("../publication_compare/base_files/WT_3dpg_COT",        216,"../publication_compare/inference/inf_WT_3dpg_COT/")
# quickerator("../publication_compare/base_files/WT_8dpg_TRUE",       216,"../publication_compare/inference/inf_WT_8dpg_TRUE/")

# for file in glob.glob("../publication_compare/base_files/AZD/*.tif"):
#     name = os.path.basename(file)
#     subprocess.run(f'python inference_SUF.py --model=checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth --input="{file}" --overlap=232 --outputdir="../publication_compare/inference/inf_AZD/" --outputname="_{os.path.splitext(name)[0]}_o232" --progress=True', shell=True)

# for file in glob.glob("../publication_compare/base_files/basl-2_5dpg_COT/*.tif"):
#     name = os.path.basename(file)
#     subprocess.run(f'python inference_SUF.py --model=checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth --input="{file}" --overlap=232 --outputdir="../publication_compare/inference/inf_basl-2_5dpg_COT/" --outputname="_{os.path.splitext(name)[0]}_o232" --progress=True', shell=True)

# for file in glob.glob("../publication_compare/base_files/TMMpLNG1_3dpg_COT/*.tif"):
#     name = os.path.basename(file)
#     subprocess.run(f'python inference_SUF.py --model=checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth --input="{file}" --overlap=232 --outputdir="../publication_compare/inference/inf_TMMpLNG1_3dpg_COT/" --outputname="_{os.path.splitext(name)[0]}_o232" --progress=True', shell=True)

# for file in glob.glob("../publication_compare/base_files/trm678_5dpg_COT/*.tif"):
#     name = os.path.basename(file)
#     subprocess.run(f'python inference_SUF.py --model=checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth --input="{file}" --overlap=232 --outputdir="../publication_compare/inference/inf_trm678_5dpg_COT/" --outputname="_{os.path.splitext(name)[0]}_o232" --progress=True', shell=True)

# for file in glob.glob("../publication_compare/base_files/UBQ10pLNG1_3dpg_COT/*.tif"):
#     name = os.path.basename(file)
#     subprocess.run(f'python inference_SUF.py --model=checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth --input="{file}" --overlap=232 --outputdir="../publication_compare/inference/inf_UBQ10pLNG1_3dpg_COT/" --outputname="_{os.path.splitext(name)[0]}_o232" --progress=True', shell=True)

# for file in glob.glob("../publication_compare/base_files/UBQ10pOFP2/*.tif"):
#     name = os.path.basename(file)
#     subprocess.run(f'python inference_SUF.py --model=checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth --input="{file}" --overlap=232 --outputdir="../publication_compare/inference/inf_UBQ10pOFP2/" --outputname="_{os.path.splitext(name)[0]}_o232" --progress=True', shell=True)

# for file in glob.glob("../publication_compare/base_files/WT_3dpg_COT/*.tif"):
#     name = os.path.basename(file)
#     subprocess.run(f'python inference_SUF.py --model=checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth --input="{file}" --overlap=232 --outputdir="../publication_compare/inference/inf_WT_3dpg_COT/" --outputname="_{os.path.splitext(name)[0]}_o232" --progress=True', shell=True)

# for file in glob.glob("../publication_compare/base_files/WT_8dpg_TRUE/*.tif"):
#     name = os.path.basename(file)
#     subprocess.run(f'python inference_SUF.py --model=checkpoints/comp_nosmalls_2025-05-21_12h-13m-25s/last.e029.pth --input="{file}" --overlap=232 --outputdir="../publication_compare/inference/inf_WT_8dpg_TRUE/" --outputname="_{os.path.splitext(name)[0]}_o232" --progress=True', shell=True)




def RGBgen_from_monodiff_vector(num, threshold=100):
    
    # Calculates the R value corresponding to the monochrome cell, to make the first image ORANGE (in combination with the G value)
    R = np.maximum( num, 0)
    # Calculates the B value corresponding to the monochrome cell, to make the first image BLUE (in combination with the G value)
    B = np.maximum(-num, 0)

    # THreshold is used to artificially inflate minor differences so that they can be seen in the first place.
    R[np.logical_and(num > 0, num <  threshold)] = threshold
    B[np.logical_and(num < 0, num > -threshold)] = threshold

    # GREEN value that scales with half the magnitude of the difference in the monochrome values.
    G = np.abs(num)//2

    # RGB array returned for saving and display with PIL.Image
    return np.stack([R,G,B], axis=-1).astype(np.uint8)


def generate_compare(img1, img2):
    # Generates an array of numbers from -255 to 255 inclusive.
    # A positive value represents a feature MORE present in the first image than the second. This will be shown in ORANGE.
    # A negative value represents a feature LESS present in the first image than in the second. This will be shown in BLUE.
    delta = np.subtract(img1.astype(np.int16), img2.astype(np.int16))
    
    # Vectorized apply to convert [-255, 255] psuedo-monochrome to ([0, 255], [0, 255], [0, 255]) RGB.
    delta_RGB = RGBgen_from_monodiff_vector(delta, threshold = 40)
    
    return delta_RGB

Image.fromarray(
    generate_compare(
        np.asarray(Image.open("inference/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged.tifCONTINGENCY.output.png")),
        np.asarray(Image.open("remove_ALL_smalls/ANNOT_smalls/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged_ANNOT.png"))
        # np.asarray(Image.open("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/inference/R3-1uM_1_1.tifo232_AZDtest_avgskiptest_withskip.output.png")),
        # np.asarray(Image.open("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/inference/R3-1uM_1_1.tifo200_AZDtest_avgskiptest_withskip.output.png"))
        )
).show()


# for i, filepair in enumerate(zip(glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/pre-annotated/AZD/*.png"), glob.glob("C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/inference/inf_AZD/raw_inf_216/*.png"))):
#     base, inf = filepair[0], filepair[1]
#     # print(base, inf)
#     subprocess.run(f"python contingency.py --ground_truth={base} --guess_image={inf} --texttag=_{i}", shell=True)


# Quantify Training and Validation loss from Models into Table/Graphs
# (done beforehand?)

### Model Stats

### Contingency

### Diff Maps

# Outside Testing - Diff Maps

### 8dpg True
### TMMpLNG1
### trm678
### UBQ10pLNG1
### UBQ10pOFP1
### basl_2

### Diff Maps

# for i in range(-255, 256):
#     print(i, RGBgen_from_monodiff(i))