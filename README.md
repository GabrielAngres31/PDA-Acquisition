

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
python inference.py --model=checkpoints/2024-02-20_12h-35m-29s/last.e029.pth \
                    --input=SCD_training_data/source_images/BASE/cot6.tif

# TODO:
python errorviz.py --ground_truth=GROUND_TRUTH  --model_predict=MODEL_PREDICT
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
