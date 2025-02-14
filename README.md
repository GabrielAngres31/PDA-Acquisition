

## Usage

```bash
#new python virtual environment
python -m venv venv

#activate it
#source venv/bin/activate   #linux
venv\Scripts\activate     #windows

#install packages
pip install -r requirements.txt

#run training 
python training.py --trainingsplit=splits/pores_only_test_02-2025.csv --validationsplit=splits/pores_only_val_02-2025.csv

#run inference on a file (adjust paths)
python inference.py --model=checkpoints/2025-02-12_19h-14m-25s/last.e029.pth --input=SCD_training_data/source_images/BASE/cot6.tif --overlap=32

# Show insufficiently confident positives in blue and false positives in orange.
# Requires two annotation files to compare, either hand-generated or machine-generated.
# Images must be BLACK background, WHITE features
python errorviz.py --ground_truth=SCD_training_data/source_images/ANNOTATION/cot6_STOMATA_MASKS.tiff  --model_predict=inference/cot6.tif.output.png --show=1

# Obtain aggregate data on non-black clump sizes

python clumps_table.py --input_path="C:\Users\Gabriel\Documents\GitHub\PDA-Acquisition\only_pored\ANNOT\basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged_ANNOT.png" --output_folder="inference" --prediction_type="clumps" --filter_type="otsu" --save_image_as="basl-2_5_04_128_ANNOT"

# Create scatterplots of measures
python visualizers.py --source_data=inference/basl-2_5_COT_02_rotated_MAX_basl-2_5dpg_110321_1_2_abaxial_merged_ANNOT_modded.csv --scatterplots="axis_major_length,axis_minor_length|eccentricity,perimeter|area,axis_minor_length|area,axis_major_length" --save_as="basl2-5-02_ANNOT_glance"

# Create ridgeplots of measures
# YOU MUST HAVE AN ID COLUMN - data comes from multiple leaves
python visualizers.py --source_data=inference/clump_data/aggregate_folders/aggregate.csv --ridgeplots="area,axis_major_length,axis_minor_length,eccentricity,extent"

# Create histograms of measures
python visualizers.py --source_data=inference/clump_data/cot6_STOMATA_MASKS.csv --histograms="area,axis_major_length,axis_minor_length,eccentricity"
```


## Code Overviews

### Data.py
DEF load_splitfile
    Read a .csv file containing paths to input images and annnotations.
DEF load_image
    Loads an image.
DEF load_inputimage (unused?)
    USE load_image with preconfigured parameters for base image files.
DEF load_annotation
    USE load_image with preconfigured parameters for annotation files.
DEF save_image
    Saves an image.

DEF cache_filepairs
    USE load_image
    USE load_annotation
