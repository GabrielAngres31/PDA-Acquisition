

## Usage

```bash
# Change directory
cd "C:/Users/[USER]/Downloads/install_standalone"

#new python virtual environment
python -m venv venv

#activate it

#source venv/bin/activate   #linux
venv\Scripts\activate     #windows

#install packages
pip install -r requirements.txt

# Clean an image on a threshold ( [0-1) ) with Otsu, or on only an Otsu threshold
python clean_image_SUF.py --input_path="C:\Users\Gabriel\Documents\GitHub\PDA-Acquisition\only_pored\ANNOT\basl-2_5_COT_04_rotated_MAX_basl-2_5dpg_110321_2_2_abaxial_merged_ANNOT.png" --filter_type="otsu" --save_image_as="basl-2_5_04_128_ANNOT"

# Obtain aggregate data on non-black clump sizes
python clumps_table_SUF.py --input_path="cleaned_images_default/basl-2_5_04_128_ANNOT.png" --output_folder="inference"

# Boot annotation helper
python tkinter_GUI_test_SUF.py