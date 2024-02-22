

## Usage

```bash
#new python virtual environment
python -m venv venv

#install packages
pip install -r requirements.txt

#run training 
python training.py --trainingsplit=splits/train_0.csv

#run inference on a file (adjust paths)
python inference.py --model=checkpoints/2024-02-20_12h-35m-29s/last.e029.pth \
                    --input=SCD_training_data/source_images/BASE/cot6.tif
```

