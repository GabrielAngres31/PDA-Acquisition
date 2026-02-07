

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

python training.py --trainingsplit="nosmallsplitval_05-2025.csv" --patchsize=128 --overlap=16 --outputcsv="mini_nonoise_p128_o16" --epochs=30

python inference.py --model=install_standalone/model/2025-02-12_19h-14m-25s/last.e029.pth --input="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/SCD_training_data/source_images/BASE/cotE12.tif" --overlap=128 --outputname="__PUB-WT5COTE12.png" --outputdir="Documents/" --skip_empty=True --progress=True

python inference_mbn.py --model="install_standalone/checkpoints/mbn/2025-07-30_12h-05m-34s/last.e029.pth" --input_folder=C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/training_folder_clumpcatcher/val/single

python training_mbn.py --trainingfolder="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/training_folder_clumpcatcher/train/" --validationfolder="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/training_folder_clumpcatcher/val/"  --outputcsv="testtable_retest.csv"
```

### Image Cleaning and Manipulation
```bash
# Image Cleaner
#   Clean an image on a threshold ( [0-1) ) with Otsu, or on only an Otsu threshold
# TODO: Determine if clean_image.py or imagecleaner.py is the most recent/useful one
python clean_image.py --input_path="C:\Users\Gabriel\Documents\GitHub\PDA-Acquisition\only_pored\ANNOT\basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged_ANNOT.png" --filter_type="otsu" --save_image_as="basl-2_5_04_128_ANNOT"

# Stomata Splitter
#   generate centered images of stomata based on a cleaned annotation for pair-calling training or inference.
# TODO: commandify the splitter, remove magic numbers
python stomatasplitter.py

```

### Statistics
```bash
# Clump Finder
#   obtains aggregate data on non-black clump sizes
python clumps_table.py --input_path="cleaned_images_default/basl-2_5_04_128_ANNOT.png" --output_folder="inference"

# Contingency Calculator
#   

python contingency.py --ground_truth="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/mbn_scrapbox_folder/new_img_annot/CLN__cotE05.png" --guess_image="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/inference/BASE_cotE05_BLEANED.png" --show_image=True --output_folder_table=. --texttag="PUBFIND"

# Contrast Mapper
#   
python contrastmap.py --base_img="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/install_standalone/mbn_scrapbox_folder/new_img_annot/CLN__cotE05.png" --compare_img="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/inference/BASE__cotE05_BLASTED.png" --threshold=0 --output_path="C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/publication_compare/contingency/images/cot6_64" --show_image=True

# TODO: Find out what margin_pixel_calc.py does
```