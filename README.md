

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
python training.py --trainingsplit=splits/train_0.csv

#run inference on a file (adjust paths)
python inference.py --model=checkpoints/2024-06-27_11h-31m-26s/last.e029.pth --input=SCD_training_data/source_images/BASE/cot6.tif --overlap=32

#show insufficiently confident positives in blue and false positives in orange
python errorviz.py --ground_truth=SCD_training_data/source_images/ANNOTATION/cot6_STOMATA_MASKS.tiff  --model_predict=inference/cot6.tif.output.png --show=1

#obtain aggregate data on non-black clump sizes
# NOTE: Will be replaced by clumps_table.py which generates the table and a separate library of helper functions to generate the figures.
python clumpfinder.py --input_path=inference/cot6.tif.output.png --closing_threshold=80 --opening_threshold=120 --scatter_plot=1 --area_histogram=1
python clumpfinder.py --input_path=source_images/ANNOTATION/cot6_STOMATA_MASKS.tif --closing_threshold=80 --opening_threshold=120 --scatter_plot=1 --area_histogram=1
python clumpfinder.py --input_path=inference/cot6.tif.output.png --closing_threshold=80 --opening_threshold=120 --properties="label,area,axis_major_length,axis_minor_length,centroid,eccentricity,extent,bbox"

# Create ridgeplot of measures
python visualizers.py --source_data=inference/clump_data/aggregate.csv --ridgeplots="area,axis_major_length,axis_minor_length,eccentricity,extent"
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
