import os
import subprocess

OUT_tst_mdl_path = "outline_test_only"
OUT_val_mdl_path = "outline_val"

BLB_tst_mdl_path = "blobs_test_only"
BLB_val_mdl_path = "blobs_val"


command_testing_blobs    = f"python training.py --trainingsplit=splits/train_1a.csv --checkpointdir=./checkpoints/{BLB_tst_mdl_path}/"
command_validation_blobs = f"python training.py --trainingsplit=splits/train_1a.csv --validationsplit=splits/valid_1a.csv --checkpointdir=./checkpoints/{BLB_val_mdl_path}/"
command_testing_outln    = f"python training.py --trainingsplit=splits/outline_train_0.csv --checkpointdir=./checkpoints/{OUT_tst_mdl_path}/"
command_validation_outln = f"python training.py --trainingsplit=splits/outline_train_0.csv --validationsplit=splits/outline_valid_0.csv --checkpointdir=./checkpoints/{OUT_val_mdl_path}/"


