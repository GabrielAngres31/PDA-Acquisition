from collections import deque

import torch
import torchvision

import typing as tp

import argparse

import src.data
import tqdm
import numpy
import PIL

import matplotlib.pyplot as plt

import skimage.measure as skimm
import skimage.morphology as skimorph

#import pandas as pd


def main(args:argparse.Namespace) -> bool:
    
    target_tensor = src.data.load_image(args.input_path, "L")
    
    table = find_clumps_skimage(target_tensor[0], args.closing_threshold, args.opening_threshold)
    
    if args.area_histogram:
    
        # plt.hist(clump_info_dict.values(), bins=list(range(0, 1000, 50)))
        # plt.title("Clump Sizes")
        # plt.show()
        #print(clump_info_dict)
        assert 'area' in table.keys(), "Area is not listed in this table!"

        plt.hist(table["area"], bins=list(range(0, 2000, 120)))
        plt.title("Clump Sizes")
        if args.save_to:
            plt.savefig(f"reference_figures/{args.save_to}_area_histogram.png")
        #plt.show()
        plt.clf()
        
        
        
    if args.scatter_plot:
        assert 'axis_major_length' in table.keys(), "Major Axis Length is not listed in this table!"
        plt.scatter(table["area"], table["axis_major_length"])
        plt.xlim(0, 2000)
        plt.ylim(0,   60)
        plt.title("Clump Sizes vs. Major Axis Length")
        if args.save_to:
            plt.savefig(f"reference_figures/{args.save_to}_area_axis_scatter.png")
        #plt.show()
        plt.clf()
        
        heatmap, xedges, yedges = numpy.histogram2d(table["area"], table["axis_major_length"]*1000, bins = [list(range(0, 2000, 50)), list(range(0, 1000, 50))])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        plt.clf()
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        if args.save_to:
            plt.savefig(f"reference_figures/{args.save_to}_area_axis_heatmap.png")
        #plt.show()
        plt.clf()
        
    return True

def find_clumps_skimage(image: PIL.Image, closing_threshold: int, opening_threshold: int, properties: tuple = None): #-> None

    # image_numpy = skimorph.area_opening(image, area_threshold = 200)
    image = (image > 0.1)
    image_numpy = numpy.asarray(image)

    #src.data.save_image("example.png", numpy.asarray(image).astype(numpy.float32))
    #print(closing_threshold)
    image_closing = skimorph.area_closing(image_numpy, area_threshold = closing_threshold) #80
    image_opening = skimorph.area_opening(image_closing, area_threshold = opening_threshold) #120

    src.data.save_image("reference_figures/example_both.png", image_opening.astype(numpy.float32))

    clumps_map = skimm.label(image_opening, connectivity=2) 
    shape = clumps_map.shape
    #plt.imshow(clumps_map)
    #plt.show()
    #print(shape)
    #plt.clf()
    if properties:
        props = properties
    else:
        props = ('label', 'bbox')
        # ('label', 'bbox', 'area', 'area_bbox', 'axis_major_length', 'axis_minor_length', 'centroid', 'eccentricity', 'area_convex', 'perimeter')
    return skimm.regionprops_table(skimm.label(clumps_map), properties = props) #, 'eccentricity'

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type = str,
        required = True,
        help = 'Image to find chunks for.'
    )
    parser.add_argument(
        '--area_histogram',
        type = int,
        default = 0,
        help = 'Whether or not to generate a histogram of clump sizes.'
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
    # parser.add_argument(
    #     '--confidence_threshold',
    #     type = float,
    #     required = True,
    #     help = "Threshold below which pixels of a given confidence are not counted in the final clump."
    # )
    parser.add_argument(
        '--scatter_plot',
        type = int,
        help = "Whether or not to generate a scatter plot of clump sizes versus eccentricity."
    )
    parser.add_argument(
        '--save_to',
        type = str,
        help = "What identifier to save files with. Leave blank to avoid saving files."
    )
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')