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
    
    # CLUMPS-based image workflow
    # target_tensor = src.data.load_image(args.input_path, "L")
    # table = find_clumps_skimage(target_tensor[0])
    image_in = skimage.io.imread(args.input_path)
    table = find_clumps_skimage(image_in)

    # Outline-based image workflow:
    
    
    pd.DataFrame(table).to_csv(f"{args.output_folder}/{Path(args.input_path).stem}.csv")

def find_clumps_skimage(image: PIL.Image): #-> None
    otsu_trsh_num = threshold_otsu(image)
    otsu_fill = area_closing(image > otsu_trsh_num, connectivity = square(3), area_threshold = 2500) # Hardcoded dark patch value
    otsu_invt = skimage.util.invert(image > otsu_trsh_num)
    

    inners = np.logical_and(otsu_fill, otsu_invt)
    otsu_clr = area_opening(inners, area_threshold = 200)
    

    ### FORMER IMAGE PROCESSING STEPS FOR RAW IMAGES/CLUMPS

    # image = (image > 0.1)
    # image_numpy = numpy.asarray(image)

    # image_closing = skimorph.area_closing(image_numpy, area_threshold = closing_threshold) #80
    # image_opening = skimorph.area_opening(image_closing, area_threshold = opening_threshold) #120

    

    clumps_map = skimm.label(otsu_clr, connectivity=2) 
    # shape = clumps_map.shape

    #return skimm.regionprops_table(skimm.label(clumps_map)) #, 'eccentricity'
    return skimm.regionprops_table(skimm.label(clumps_map), 
                                   properties = ('label', 
                                                 'bbox', 
                                                 'area', 
                                                 'area_bbox', 
                                                 'axis_major_length', 
                                                 'axis_minor_length', 
                                                 'centroid', 
                                                 'eccentricity', 
                                                 'area_convex', 
                                                 'perimeter',
                                                 'equivalent_diameter_area',
                                                 'extent',
                                                 'orientation')
                                                 ) #, 'eccentricity'

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type = str,
        required = True,
        help = 'Image to find chunks for.'
    )
    parser.add_argument(
        '--output_folder',
        type = str,
        required = True,
        help = 'Which folder to store the output table in.'
    )
    # parser.add_argument(
    #     '--closing_threshold',
    #     type = int,
    #     required = True,
    #     help = "Threshold to remove dark holes in stomata."
    # )
    # parser.add_argument(
    #     '--opening_threshold',
    #     type = int,
    #     required = True,
    #     help = "Threshold below which clumps are not counted in the final list (conducted after closing step)."
    # )
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')
