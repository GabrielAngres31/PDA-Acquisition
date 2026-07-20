

# Usage

## Setup

Navigate to the project root directory and set up the isolated Python environment:

```bash
# Change directory
cd "C:/Users/[USER]/Downloads/install_standalone_v2"

# Create new python virtual environment
python -m venv venv                              

# Activate virtual environment
# Linux/macOS:
# source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install required packages
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

"""
  --trainingsplit       Path to training CSV image pairs
  --validationsplit     Path to validation CSV image pairs 
  --patchsize           Sidelength of input patches in pixels
  --cachedir            Folder to store image patches
  --checkpointdir       Folder to store trained models
  --overlap             Pixel amount of overlap between patches
  --outputcsv           Filename to store csv of losses and epochs
  --epochs              Number of training epochs
  --lr                  Learning rate for AdamW optimizer
  --batchsize           Number of samples in a batch per training step
  --pos_weight          Extra training weight on the positive class
"""
python training.py --trainingsplit "publication_files/splits/nosmallsplitval_06-2025.csv" --patchsize 128 --overlap 16 --outputcsv "/demo_split_128-16_07172026" --epochs 1

"""
  --model               Path to saved model
  --input               Path to input image
  --close_on            Maximum hole size to remove
  --open_on             Minimum clump size to keep
  --patchsize           Patch size, default 256
  --overlap             Vertical overlap between patches
  --outputdir           Path to outputfolder
  --outputname          Optional suffix before file type
  --progress            Whether to show a progress bar. May disable for batch jobs.
  --weights_only        
  --skip_empty          Whether the sliding window skips frames that are "empty" (border pixel average >1/255)
"""
python inference.py --model="checkpoints/model/2025-02-12_19h-14m-25s/last.e029.pth" --input="publication_files/image_data/BASE_smalls/cotE10.tif" --overlap=128 --outputname="__DEMO-COTE10.png" --outputdir="install_standalone_v2/DEMO_OUT/" --skip_empty=True --progress=True   

"""
  --trainingfolder      Path to folder containing class-labeled images to be used for training
  --validationfolder    Path to folder containing class-labeled images to be used for validation
  --checkpointdir       Where to store trained models
  --outputcsv           Name to store csv of losses and epochs for easy viewing
  --epochs              Number of training epochs
  --lr                  Learning rate for AdamW optimizer
  --batchsize           Number of samples in a batch per training step
  --pos_weight          Extra training weight on the positive class
"""
python training_mbn.py --trainingfolder="training_folder_clumpcatcher/train" --validationfolder="training_folder_clumpcatcher/val"  --outputcsv="DEMO_testtable.csv" --epochs=2 --checkpointdir="checkpoints/mbn/"


"""  
  --model               Path to saved model
  --input_image         Path to input image
  --input_folder        Path to folder with input images
  --outputdir           Path to outputfolder
  --outputname          Optional suffix before file type
"""
python inference_mbn.py --model="checkpoints/mbn/2026-07-17_14h-27m-35s/last.e001.pth" --input_folder=training_folder_clumpcatcher/val/single



"""
  --model MODEL         Path to model to validate.
  --validationsplit     Path to .csv file containing pairs of input images and annotations to be used for validation
  --patchsize           Size of input patches in pixels
  --cachedir            Where to store image patches
  --overlap             How much overlap between patches
  --batchsize           Number of samples in a batch per training step
  --pos_weight          Extra training weight on the positive class
"""
python freestanding_val.py --model=checkpoints/model/2025-02-12_19h-14m-25s/last.e029.pth --validationsplit="publication_files/splits/nosmallsplitval_06-2025.csv"
```

### Image Cleaning and Manipulation
```bash


# Image Cleaner
#   Clean an image on a threshold ( [0-1) ) with Otsu, or on only an Otsu threshold
"""
  --input_path          Image to find chunks for.
  --prediction_type     Whether the mask being read is an outline of the stomata or the stomata themselves. {clumps,outlines}
  --filter_type         Whether to filter on absolute pixel brightness or an otsu threshold. {confidence,otsu}
  --save_image_as       Optional - saves an image of the cleaned image to a directory/name.jpg you specify
"""
python clean_image.py --input_path="publication_files/image_data/ANNOT_smalls/basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged_ANNOT.png" --filter_type="otsu" --save_image_as="DEMO_cleanimage"


# Stomata Splitter
#   generate centered images of stomata based on a cleaned annotation for pair-calling training or inference.
#   Uses as input paired images of BASE and clean ANNOT files and their associated CLUMP files as generated by the Clump Finder script (below).
#   Data is an input for single/cluster calling.
"""
  --grab_list           Path to the CSV list of files to parse                          (default: grabfile.csv)
  --output_dir          Directory to save extracted crops                               (default: training_folder_clumpcatcher)
  --additional_dir      Directory containing augmented data                             (default: additional_clump_images)
  --crop_size           Total width/height of the square crop                           (default: 72)
  --split_ratio         Every Nth image goes to validation to achieve an N-1/1 split    (default: 5)
  --keep_existing       Set this flag to prevent clearing the output folder before running
"""
python stomatasplitter_cli.py --grab_list="grabfile.csv" --output_dir="DEMO_clumpcatch" 


# NOTE:
# image_audit_canvas.py is not meant to be directly run, but is called by tkinter_GUI.py.
```

### Statistics
```bash


# Clump Finder
#   obtains aggregate data on non-black clump sizes
"""
  --input_path          Image to find chunks for.
  --properties          Desired properties to calculate for each clump.
  --output_folder       Which folder to store the output table in.
"""
python clumps_table.py --input_path="publication_files/image_data/ANNOT_smalls/cotE10_ANNOT.png" --output_folder="DEMO_OUT"


# Contingency Calculator
"""
  --ground_truth        Filepath for the Ground Truth Image
  --guess_image         Annotation generated from machine inference
  --show_image          Set to see the histogram
  --texttag             Label for output histogram
  --output_folder_table Output folder
"""
python contingency_plus.py --ground_truth="publication_files/image_data/ANNOT_smalls/cotE10_ANNOT.png" --guess_image="output_folder/DEMO_E10_Cleaned.png" --show_image --output_folder_table="DEMO_OUT" --texttag="DEMO_CONTINGENT"


# Contrast Mapper
"""
  --base_img                Path to base image.
  --compare_img             Path to comparison image.
  --threshold               Absolute value threshold below which everything is given the shaded threshold value. Set to 0 to show all absolute differences. Set to 255 to highlight all pixels that do not exactly match, at maximum brightness. Set to an intermediate value to arbitratily visualize small differences.
  --output_path             Output file to save image.
  --show_image              Whether to display the image using Image imshow().
  --concordance_highlight   Whether to recolor areas of perfect concordance as pink.
"""
python contrastmap.py --base_img="publication_files/image_data/ANNOT_smalls/cotE10_ANNOT.png" --compare_img="DEMO_OUT/cotE10.tif__DEMO-COTE10.png.output.png" --threshold=0 --output_path="DEMO_OUT/DEMO_contrast.png" --show_image=True


# Margin Pixel Calculator
"""
  --input_path           Image to profile margin pixels for.
  --compare_path         Image to compare against.
"""
python margin_pixel_calc.py --input_path="publication_files/image_data/ANNOT_smalls/cotE10_ANNOT.png" --compare_path="DEMO_OUT/cotE10.tif__DEMO-COTE10.png.output.png"
```


# OTHER
```bash

# Stomata_splitter.py was run by itself and requires grabfile.csv.
# This file is not necessary to run any analyses but is kept for reproducibility.
```


# Workflow Guide
Note: Please direct any questions, requests for clarifications, or suggestions to the owner of this repository.

## Basic Analysis

Start with monochrome (black and white) images of the specimens to be analyzed.
1) Using a MODEL (.pth file) of your choice, use `inference.py` to generate unpolished annotations (ROUGHS).
2) Polish the ROUGHS using `clean_image.py` to generate crisp monochrome annotations (SMOOTHS).
3) FINAL or SMOOTH annotations may be quantified using `clump_finder.py` to produce CLUMP files. FINAL files, being more carefully examined, generally produce better CLUMP data.
4) SMOOTH files with accompanying CLUMP data may be polished in the `tkinter` gui to produce pixel-perfect annotations (FINALS)
5) FINALS and CLUMPS may be used to generate collections of centered images of stomata using the `stomata splitter`, to be evaluated by the `paircaller` as "single" or "clustered" stomata, using `inference_mbn.py`.

## Further Analysis
1) The `contingency` calculator can be used to numerically quantify the differences between SMOOTH or FINAL images produced by different models or hand-editing.
2) The `contrast mapper` can be used to __visualize__ the differences in prediction between SMOOTH or FINAL images.
3) The `margin pixel calculator` can be used to get a rough sense of the amount of error that can be ignored in a given image. Since the accuracy of all predictions have a minimum error of one pixel, by counting all the pixels on the margins of all identified clumps, an approximate quantity of a lower bound for the smallest significant discrepancy may be obtained.

## Training
1) Using a collection of paired base images and FINALS, `training.py` may be used to generate a model using the base weights. To trade between training speed and resulting model accuracy (inversely related), modify the parameters above. 
    * Increase batch size, dataset size, epochs, or overlap to increase accuracy, while decreasing the learning rate for the same purpose. 
    * Positive weight on the training class by default should not be increased unless your data contains indistinct features. 
2) Collections of centered stomata generated by the `stomata splitter` can be used to train a new `paircaller` model using `training_mbn.py`.