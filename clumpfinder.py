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

import pandas as pd


def main(args:argparse.Namespace) -> bool:
    
    target_tensor = src.data.load_image(args.input_path, "L")
    #properties = args.properties.split(":")
    properties = set(args.histogram.split("%")) | set(y for sublist in args.scatter_plot.split(":") for y in sublist.split("^"))
    properties.remove('')
    properties = list(properties)

    if args.distances:
        properties.append("centroid")
    table = find_clumps_skimage(target_tensor[0], args.closing_threshold, args.opening_threshold, save_to=args.save_to, properties = properties)

    # TODO: FIGURE OUT How to store the results of O(n^2) distances to a file specific to an image being analyzed

    if args.histogram:
        for param in args.histogram.split("%"):
        # plt.hist(clump_info_dict.values(), bins=list(range(0, 1000, 50)))
        # plt.title("Clump Sizes")
        # plt.show()
        #print(clump_info_dict)
        #assert 'area' in table.keys(), "Area is not listed in this table!"

            plt.hist(table[param])#, bins=list(range(0, 2000, 120)))
            plt.title(f"{param}")
            plt.figure(figsize = (8,6))
            if args.save_to:
                plt.savefig(f"reference_figures/{args.save_to}_area_histogram.png")
            #plt.show()
            plt.clf()
        
        
        
    if args.scatter_plot:
        for param in args.scatter_plot.split("%"):
            #assert 'axis_major_length' in table.keys(), "Major Axis Length is not listed in this table!"
            field1, field2 = param.split("^")
            plt.figure(figsize = (8,6))
            plt.scatter(table[field1], table[field2])
            # plt.xlim(0, 2000)
            # plt.ylim(0,   60)
            plt.title(f"{field1} vs. {field2}")
            
            if args.save_to:
                plt.savefig(f"reference_figures/{args.save_to}_{field1}_vs_{field2}_scatter.png")
            else:
                plt.show()
            #plt.show()
            plt.clf()
            
            heatmap, xedges, yedges = numpy.histogram2d(table[field1], table[field2])#, bins = [list(range(0, 2000, 50)), list(range(0, 1000, 50))])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            plt.clf()
            plt.imshow(heatmap.T, extent=extent, origin='lower')
            if args.save_to:
                plt.savefig(f"reference_figures/{args.save_to}_{field1}_vs_{field2}_heatmap.png")
            else:
                plt.show()
            #plt.show()
            plt.clf()
    
    if args.distances:
        #print(table)
        #print(table['centroid-0'])
        #print(table['centroid-1'])

        points = [(x,y) for x in table['centroid-0'] for y in table['centroid-1']]
        

    return True

def find_clumps_skimage(image: PIL.Image, closing_threshold: int, opening_threshold: int, save_to: str, properties: tuple = (" ")): #-> None
    if not properties:
        properties = ["label"]
    assert properties, "You haven't put any properties!"
    print(f"Properties: {properties}")
    # image_numpy = skimorph.area_opening(image, area_threshold = 200)
    image = (image > 0.1)
    image_numpy = numpy.asarray(image)

    

    #src.data.save_image("example.png", numpy.asarray(image).astype(numpy.float32))
    #print(closing_threshold)
    image_closing = skimorph.area_closing(image_numpy, area_threshold = closing_threshold) #80
    image_opening = skimorph.area_opening(image_closing, area_threshold = opening_threshold) #120

    if save_to:
        src.data.save_image(f"reference_figures/{save_to}.png", image_opening.astype(numpy.float32))

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
    # parser.add_argument(
    #     '--properties',
    #     type = str,
    #     required = False,
    #     help = 'List of properties separated by commas.'
    # )
    parser.add_argument(
        '--histogram',
        type = str,
        default = "",
        help = 'List of parameters you would like to plot in a histogram, separated by colons.'
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
        type = str,
        default="",
        help = "Whether or not to generate scatter plots, separated by colons and commas"
    )
    parser.add_argument(
        '--distances',
        type = int,
        default=0,
        help = "Whether or not to generate a histogram of interstomatal distances."
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