# clumps_table.py

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
import skimage
import numpy as np 

def main(args:argparse.Namespace) -> bool:
    
    image_in = skimage.io.imread(args.input_path)

    # Force 2D array from 3D
    if len(image_in.shape) == 3:
        image_in = image_in[:, :, 0]
    table = quantify_clumps_skimage(image_in,  args.properties, args.output_folder)
    
    pd.DataFrame(table).to_csv(f"{args.output_folder}/{Path(args.input_path).stem}.csv")

def quantify_clumps_skimage(image: Image, properties:tuple, saveas: str): #-> None

    clumps_map = skimage.measure.label(image, connectivity=2) 
    clumps_table = skimage.measure.regionprops_table(skimage.measure.label(clumps_map), properties = tuple(properties))
    if saveas:
        pd.DataFrame(clumps_table).to_csv(saveas + "/table_out.csv")
    return clumps_table

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
        default = ['label', 'bbox', 'area', 'area_bbox', 'axis_major_length', 'axis_minor_length', 'centroid', 'eccentricity', 'area_convex', 'perimeter', 'equivalent_diameter_area', 'extent', 'orientation'],
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
