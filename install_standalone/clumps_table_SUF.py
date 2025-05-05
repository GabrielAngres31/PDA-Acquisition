# clumps_table.py

import pandas as pd
import argparse
import src.data
import tqdm
import numpy
import PIL
from pathlib import Path

import skimage.measure as skimm
import skimage.morphology as skimorph

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, area_closing, area_opening
from skimage.color import label2rgb
import PIL
import numpy as np

def main(args:argparse.Namespace) -> bool:
    
    image_in = skimage.io.imread(args.input_path)

    # Force 2D array from 3D
    if len(image_in.shape) == 3:
        image_in = image_in[:, :, 0]
    table = quantify_clumps_skimage(image_in,  args.properties, args.output_folder)
    
    pd.DataFrame(table).to_csv(f"{args.output_folder}/{Path(args.input_path).stem}.csv")

def quantify_clumps_skimage(image: PIL.Image, properties:tuple, saveas: str): #-> None

    clumps_map = skimm.label(image, connectivity=2) 
    return skimm.regionprops_table(skimm.label(clumps_map),#, 
                                   properties = tuple(properties)
                                   )

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type = str,
        required = True,
        help = 'Image to find chunks for.'
    )
    parser.add_argument(
        '--properties',
        type = str,
        nargs = "+",
        default = ['label', 'bbox', 'area', 'area_bbox', 'axis_major_length', 'axis_minor_length', 'centroid', 'eccentricity', 'area_convex', 'perimeter', 'equivalent_diameter_area', 'extent', 'orientation'], # Eccentricity is bugged, so it's been excluded
        help = 'Desired properties to calculate for each clump.'
    )
    parser.add_argument(
        '--output_folder',
        type = str,
        required = True,
        help = 'Which folder to store the output table in.'
    )
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')
