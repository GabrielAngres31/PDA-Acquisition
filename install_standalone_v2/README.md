

# Usage

## Setup

```bash
# Change directory
cd "C:/Users/[USER]/Downloads/install_standalone_v1"

# Create new python virtual environment
python -m venv venv

# Activate virtual environment

#source venv/bin/activate   #linux
venv\Scripts\activate     #windows

# Install packages to environment
pip install -r requirements.txt
```

## Run GUI
```bash
# Boot annotation helper
python tkinter_GUI.py
```

## Manual Commands

### Training and Inference
```bash
# TODO: Training/inference for segmentation and mbn/paircaller

 # CHECKED!
python training.py --trainingsplit="publication_files/splits/nosmallsplitval_06-2025.csv" --patchsize=128 --overlap=16 --outputcsv="/demo_split_128-16_07172026" --epochs=1

 # CHECKED!
python inference.py --model="checkpoints/2026-07-17_13h-25m-52s/last.e000.pth" --input="publication_files/image_data/BASE_smalls/cotE10.tif" --overlap=128 --outputname="__DEMO-COTE10.png" --outputdir="install_standalone_v2/DEMO_OUT/" --skip_empty=True --progress=True   

 # CHECKED!
python training_mbn.py --trainingfolder="training_folder_clumpcatcher/train" --validationfolder="training_folder_clumpcatcher/val"  --outputcsv="DEMO_testtable.csv" --epochs=2 --checkpointdir="checkpoints/mbn/"

 # CHECKED!
python inference_mbn.py --model="checkpoints/mbn/2026-07-17_14h-27m-35s/last.e001.pth" --input_folder=training_folder_clumpcatcher/val/single

 # CHECKED!
python freestanding_val.py --model=checkpoints/model/2025-02-12_19h-14m-25s/last.e029.pth --validationsplit="publication_files/splits/nosmallsplitval_06-2025.csv"
```

### Image Cleaning and Manipulation
```bash

 # CHECKED!
# Image Cleaner
#   Clean an image on a threshold ( [0-1) ) with Otsu, or on only an Otsu threshold
python clean_image.py --input_path="publication_files/image_data/ANNOT_smalls/basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged_ANNOT.png" --filter_type="otsu" --save_image_as="DEMO_cleanimage"

# VERIFYING
# Stomata Splitter
#   generate centered images of stomata based on a cleaned annotation for pair-calling training or inference.
# TODO: commandify the splitter, remove magic numbers
python stomatasplitter.py


# NOTE:
# image_audit_canvas.py is not meant to be directly run, but is called by tkinter_GUI.py.
```

### Statistics
```bash
# Clump Finder
#   obtains aggregate data on non-black clump sizes
python clumps_table.py --input_path="cleaned_images_default/basl-2_5_04_128_ANNOT.png" --output_folder="inference"

# Contingency Calculator
#   
python contingency_plus.py --ground_truth="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/mbn_scrapbox_folder/new_img_annot/CLN__cotE05.png" --guess_image="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/inference/BASE_cotE05_BLEANED.png" --show_image=True --output_folder_table=. --texttag="PUBFIND"

# Contrast Mapper
#   
python contrastmap.py --base_img="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/mbn_scrapbox_folder/new_img_annot/CLN__cotE05.png" --compare_img="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/inference/BASE__cotE05_BLASTED.png" --threshold=0 --output_path="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/contingency/images/cot6_64" --show_image=True

# Margin Pixel Calculator
# 
python margin_pixel_calc.py --input_path=C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/hitsandmisses/annot/cot6_ANNOT.png --compare_path=c:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/hitsandmisses/scan/cot6.png
```


# OTHER
```bash

# Stomata_splitter.py was run by itself and requires grabfile.csv.
# This file is not necessary to run any analyses but is kept for reproducibility.
```



# Example Workflow

