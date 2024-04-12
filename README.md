

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
python inference.py --model=checkpoints/2024-04-09_11h-56m-29s/last.e029.pth --input=SCD_training_data/source_images/BASE/cot6.tif

# TODO:
python errorviz.py --ground_truth=SCD_training_data/source_images/ANNOTATION/cot6_STOMATA_MASKS.tiff  --model_predict=inference/cot6.tif.output.png --show=1
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
