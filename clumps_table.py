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
    table = find_clumps_skimage(image_in, args.prediction_type, args.filter_type, args.save_image_as)

    # Outline-based image workflow:
    
    
    pd.DataFrame(table).to_csv(f"{args.output_folder}/{Path(args.input_path).stem}.csv")

def find_clumps_skimage(image: PIL.Image, mode:str, filter_mode:str, saveas: str): #-> None
    #image = skimage.color.rgb2gray(image)
    if len(image.shape) == 3:
        image = skimage.color.rgb2gray(image)
    print(image.shape)
    if filter_mode == "otsu":
        if mode=="outlines":
            otsu_trsh_num = threshold_otsu(image)
            otsu_fill = area_closing(image > otsu_trsh_num, connectivity = square(3), area_threshold = 2500) # Hardcoded dark patch value
            otsu_invt = skimage.util.invert(image > otsu_trsh_num)

            inners = np.logical_and(otsu_fill, otsu_invt)
            otsu_clr = area_opening(inners, area_threshold = 200)
            final_image=otsu_clr
        elif mode=="clumps":
            
            #image = (image/255 > 0.95)
            # print(image)
            print("Thresholding...")
            thresh_value = threshold_otsu(image)
            thresh_image = (image>thresh_value)
            image_numpy = numpy.asarray(thresh_image)


            print("Closing...")
            image_closing = skimorph.area_closing(image_numpy, area_threshold = 0) #80
            print("Opening...")
            image_opening = skimorph.area_opening(image_closing, area_threshold = 0) #120
            final_image=image_opening
    elif filter_mode == "confidence":
        final_image = image/255 > 0.87


    if saveas:
        src.data.save_image(f"reference_figures/{saveas}.png", final_image.astype(numpy.float32))

    clumps_map = skimm.label(final_image, connectivity=2) 
    # shape = clumps_map.shape
    print("returning")
    #return skimm.regionprops_table(skimm.label(clumps_map)) #, 'eccentricity'
    return skimm.regionprops_table(skimm.label(clumps_map),#, 
                                   properties = ('label',#, 
                                                 'bbox', 
                                                 'area', 
                                                 #'area_bbox', 
                                                 'axis_major_length', 
                                                 'axis_minor_length', 
                                                 #'centroid', 
                                                 'eccentricity', 
                                                 'area_convex', 
                                                 'perimeter',
                                                 #'equivalent_diameter_area',
                                                 #'extent',
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
        '--prediction_type',
        type = str,
        required = True,
        choices = ['clumps', 'outlines'],
        help = 'Whether the mask being read is an outline of the stomata or the stomata themselves.'
    )
    parser.add_argument(
        '--filter_type',
        type = str,
        required = True,
        choices = ['confidence', 'otsu'],
        help = 'Whether to filter on absolute pixel brightness or the otsu threshold.'
    )
    parser.add_argument( 
        '--save_image_as',
        type = str,
        help = 'Optional - saves an image of the cleaned image to a directory/name.jpg you specify'
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
