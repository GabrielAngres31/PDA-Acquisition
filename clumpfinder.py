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

import pandas as pd


def main(args:argparse.Namespace) -> bool:
    
    target_tensor = src.data.load_image(args.input_path, "L")

    #clumps_list = find_clumps(target_tensor, args.size_threshold, args.confidence_threshold)
    print(target_tensor)
    table = find_clumps_skimage(target_tensor[0])
    #print(table)
    
    if args.area_histogram:
        # pass
        # clump_info_dict:tp.Dict[int, int] = {}
        # for id in clumps_list.keys():
        #     #print(id)
        #     #confidence_values = []
        #     #for pixel in clumps_list[id]:
        #     #    #print(pixel)
        #     #    #print(target_tensor[0, pixel[0], pixel[1]])
        #     #    confidence_values.append(target_tensor[0, pixel[0], pixel[1]])
        #     #clump_info_dict[id] = (len(clumps_list[id]), confidence_values)
        #     clump_info_dict[id] = len(clumps_list[id])
    
        # plt.hist(clump_info_dict.values(), bins=list(range(0, 1000, 50)))
        # plt.title("Clump Sizes")
        # plt.show()
        #print(clump_info_dict)
        assert 'area' in table.keys(), "Area is not listed in this table!"

        plt.hist(table["area"], bins=list(range(0, 2000, 120)))
        plt.title("Clump Sizes")
        plt.show()
        
    if True or args.colorize:
        
        pass
        # output_colorized = src.data.load_image(args.input_path, "RGB").permute(1, 2, 0)
        # output_colorized = (output_colorized * 255).to(torch.uint8)
        # for id in clumps_list.keys():
            
        #     sample_clump = clumps_list[id]
        #     sample_clump.sort(key=lambda x: (x[0], x[1]))

        #     tag_pixel = sample_clump[0]

        #     # Color is based on pixel location within image so that the appearance or disappearance of clumps does not affect clump color.
        #     # RGB values are floored at 55 minimum to enhance visibility on dark backgrounds.
        #     color = [tag_pixel[0]%200+55, tag_pixel[1]%200+55, (tag_pixel[0]+tag_pixel[1])%200+55]
            
        #     for pixel in clumps_list[id]:
                
        #         output_colorized[pixel[0], pixel[1], 0] = color[0]
        #         output_colorized[pixel[0], pixel[1], 1] = color[1]
        #         output_colorized[pixel[0], pixel[1], 2] = color[2]

        # filename_component = args.input_path.split("/")[-1][0:-4]
        
        # src.data.save_image_RGB("inference/"+filename_component+"color.output.png", output_colorized.numpy())
        
    if args.scatter_plot:
        assert 'eccentricity' in table.keys(), "Eccentricity is not listed in this table!"
        plt.scatter(table["area"], table["eccentricity"])
        plt.title("Clump Sizes vs. Eccentricity")
        plt.show()
        # heatmap, xedges, yedges = numpy.histogram2d(table["area"], table["eccentricity"]*1000, bins = [list(range(0, 2000, 50)), list(range(0, 1000, 50))])
        # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        # plt.clf()
        # plt.imshow(heatmap.T, extent=extent, origin='lower')
        # plt.show()
        pass
    return True

def find_clumps_skimage(image: PIL.Image) -> None:
    #print(skimm.label(image, connectivity=2))
    clumps_map = skimm.label(image, connectivity=2) 
    shape = clumps_map.shape
    print(shape)
    #print([clumps_map[0, x, y] for x in range(shape[1]) for y in range(shape[2]) if clumps_map[0, x, y] > 0])
    
    #[print(x) for x in skimm.label(image, return_num=True, connectivity=2)[0][0]]

    #print(clustarrays(numpy.ndarray(skimm.label(image, connectivity=2)).keys()))
    return skimm.regionprops_table(skimm.label(image), properties = ('label', 'bbox', 'area', 'area_bbox', 'axis_major_length', 'centroid', 'eccentricity')) #, 'eccentricity'


def iterative_flood_fill(matrix: torch.Tensor, x: int, y: int, visited: torch.Tensor) -> tp.List[tp.Tuple[int, int]]:
    queue = deque()
    queue.append((x, y))
    points = list()
    use_matrix = matrix
    directions: tp.List[tp.Tuple[int, int]] = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
    while queue:
        x, y = queue.pop()
        if x < 0 or x >= use_matrix.shape[0] or y >= use_matrix.shape[1]:
            continue
        if visited[x, y] or use_matrix[x, y] == 0:
            continue

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < use_matrix.shape[0] and 0 <= ny < use_matrix.shape[1]:
                if (nx, ny) not in points:
                    queue.appendleft((nx, ny))
        visited[x, y] = True
        points.append(tuple([x,y]))
    return points

def find_clumps(matrix: torch.Tensor, size_threshold: int, confidence_threshold: float) -> tp.Dict[int, tp.List[tp.Tuple[int, int]]]:
    use_matrix = matrix[0]
    height, width = use_matrix.shape
    visited = torch.zeros_like(use_matrix, dtype=torch.bool)
    clumps: tp.Dict[int, tp.List[tp.Tuple[int, int]]] = {}
    clump_id = 0

    for x in tqdm.tqdm(range(0, height, 2)):
        for y in range(0, width, 2):
            if use_matrix[x, y] > 0 and not visited[x, y]:
                clump_points = iterative_flood_fill(use_matrix, x, y, visited)
                if clump_points:
                    confident_clump_points = [point for point in clump_points if use_matrix[point[0], point[1]] > confidence_threshold]
                    if len(clump_points) < size_threshold:
                        break
                    else:    
                        clumps[clump_id] = clump_points
                        clump_id += 1
    return clumps



def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type = str,
        required = True,
        help = 'Image to find chunks for.'
    )
    # parser.add_argument(
    #     '--colorize',
    #     type = int,
    #     default = 0,
    #     help = "Set to 1 to generate an image with the clumps colored in, and 0 otherwise."
    # )
    parser.add_argument(
        '--area_histogram',
        type = int,
        default = 0,
        help = 'Whether or not to generate a histogram of clump sizes.'
    )
    # parser.add_argument(
    #     '--density',
    #     type = int,
    #     default = 0,
    #     help = "Set to 1 to analyze relative clump size vs. clump density on a scatterplot."
    # )
    # parser.add_argument(
    #     '--size_threshold',
    #     type = int,
    #     required = True,
    #     help = "Threshold below which clumps are not counted in the final list."
    # )
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
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')