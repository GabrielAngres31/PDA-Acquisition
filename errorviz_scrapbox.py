# errorviz_scrapbox

import argparse
import os
import time
import typing as tp

import numpy as np
import torch
import torchvision
import PIL.Image

import src.data
import sys
import subprocess


def visualize_auto():
    for filename in os.listdir("SCD_training_data/source_images/BASE/basl-2_5dpg_cotyledons"):
        filehandle = filename[:-4]
        print(filehandle)
        command_vis_before = f"python errorviz.py \
        --ground_truth=SCD_training_data/source_images/ANNOTATION/{filehandle}.tif.output.png  \
        --model_predict=reference_figures/basl_predictions_compare/model_before_trm-basl/{filehandle}.tif.output.png \
        --save=reference_figures/basl_predictions_compare/BEFORE_{filehandle}.png"
        subprocess.run(command_vis_before, shell=True)
        command_vis_after = f"python errorviz.py \
        --ground_truth=SCD_training_data/source_images/ANNOTATION/{filehandle}.tif.output.png  \
        --model_predict=reference_figures/basl_predictions_compare/model_after_trm-basl/{filehandle}.tif.output.png \
        --save=reference_figures/basl_predictions_compare/AFTER_{filehandle}.png"
        subprocess.run(command_vis_after, shell=True)
        

visualize_auto()
