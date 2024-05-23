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

def main(args:argparse.Namespace) -> bool:
    
    target_tensor = src.data.load_image(args.input_path, "L")
    table = find_clumps_skimage(target_tensor[0], args.closing_threshold, args.opening_threshold)
    
    pd.DataFrame(table).to_csv(f"{args.output_folder}/{Path(args.input_path).stem}.csv")

def find_clumps_skimage(image: PIL.Image, closing_threshold: int, opening_threshold: int): #-> None
    # image_numpy = skimorph.area_opening(image, area_threshold = 200)
    image = (image > 0.1)
    image_numpy = numpy.asarray(image)

    image_closing = skimorph.area_closing(image_numpy, area_threshold = closing_threshold) #80
    image_opening = skimorph.area_opening(image_closing, area_threshold = opening_threshold) #120

    #src.data.save_image("reference_figures/example_both.png", image_opening.astype(numpy.float32))

    clumps_map = skimm.label(image_opening, connectivity=2) 
    # shape = clumps_map.shape

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
    parser.add_argument(
        '--closing_threshold',
        type = int,
        required = True,
        help = "Threshold to remove dark holes in stomata."
    )
    parser.add_argument(
        '--opening_threshold',
        type = int,
        required = True,
        help = "Threshold below which clumps are not counted in the final list (conducted after closing step)."
    )
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')
