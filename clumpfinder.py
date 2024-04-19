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

def main(args:argparse.Namespace) -> bool:
    
    target_tensor = src.data.load_image(args.input_path, "L")
    clumps_list = find_clumps(target_tensor, args.size_threshold, args.confidence_threshold)
       
    if args.histogram:
        clump_info_dict:tp.Dict[int, int] = {}
        for id in clumps_list.keys():
            print(id)
            #confidence_values = []
            #for pixel in clumps_list[id]:
            #    #print(pixel)
            #    #print(target_tensor[0, pixel[0], pixel[1]])
            #    confidence_values.append(target_tensor[0, pixel[0], pixel[1]])
            #clump_info_dict[id] = (len(clumps_list[id]), confidence_values)
            clump_info_dict[id] = len(clumps_list[id])
    
        plt.hist(clump_info_dict.values(), bins=list(range(0, 1000, 50)))
        plt.title("Clump Sizes")
        plt.show()
        print(clump_info_dict)
    
    if args.density:
        density_info:tp.Dict[int, tp.Tuple[int, int, int, float]] = {}
        for id in clumps_list.keys():
            pixels_active = len(clumps_list[id])
            x_values = sorted([x[0] for x in clumps_list[id]])
            y_values = sorted([y[1] for y in clumps_list[id]])
            x_span = x_values[-1] - x_values[0]
            y_span = y_values[-1] - y_values[0]
            total_pixels = x_span*y_span
            
            pixels_inactive = numpy.abs(total_pixels-pixels_active)
            print(f"{pixels_active}, {total_pixels}")
            density_info[id] = (total_pixels, pixels_active, pixels_inactive, (pixels_active/total_pixels if total_pixels > 0 else 0))
        list_total    = [n[0] for n in density_info.values()]
        list_active   = [n[1] for n in density_info.values()]
        list_inactive = [n[2] for n in density_info.values()]
        list_density  = [n[3] for n in density_info.values()]
        plt.scatter(list_inactive, list_active)
        plt.show()
        

    if args.colorize:
        output_colorized = src.data.load_image(args.input_path, "RGB").permute(1, 2, 0)
        output_colorized = (output_colorized * 255).to(torch.uint8)
        for id in clumps_list.keys():
            
            sample_clump = clumps_list[id]
            sample_clump.sort(key=lambda x: (x[0], x[1]))

            tag_pixel = sample_clump[0]

            # Color is based on pixel location within image so that the appearance or disappearance of clumps does not affect clump color.
            # RGB values are floored at 55 minimum to enhance visibility on dark backgrounds.
            color = [tag_pixel[0]%200+55, tag_pixel[1]%200+55, (tag_pixel[0]+tag_pixel[1])%200+55]
            
            for pixel in clumps_list[id]:
                
                output_colorized[pixel[0], pixel[1], 0] = color[0]
                output_colorized[pixel[0], pixel[1], 1] = color[1]
                output_colorized[pixel[0], pixel[1], 2] = color[2]

        filename_component = args.input_path.split("/")[-1][0:-4]
        
        src.data.save_image_RGB("inference/"+filename_component+"color.output.png", output_colorized.numpy())
        
    return True


def iterative_flood_fill(matrix: torch.Tensor, x: int, y: int, visited: torch.Tensor) -> tp.List[tp.Tuple[int, int]]:
    queue = deque()
    queue.append((x, y))
    points = list()
    use_matrix = matrix



    ### Strict Adjacency/Taxicab/NSEW: [(0,1), (1,0), (0,-1), (-1,0)]
    #directions: tp.List[tp.Tuple[int, int]] = [(0,1), (1,0), (0,-1), (-1,0)]
    
    ### Full Neighborhood: [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
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
    parser.add_argument(
        '--colorize',
        type = int,
        default = 0,
        help = "Set to 1 to generate an image with the clumps colored in, and 0 otherwise."
    )
    parser.add_argument(
        '--histogram',
        type = int,
        default = 0,
        help = 'Whether or not to generate a histogram of clump sizes.'
    )
    parser.add_argument(
        '--density',
        type = int,
        default = 0,
        help = "Set to 1 to analyze relative clump size vs. clump density on a scatterplot."
    )
    parser.add_argument(
        '--size_threshold',
        type = int,
        required = True,
        help = "Threshold below which clumps are not counted in the final list."
    )
    parser.add_argument(
        '--confidence_threshold',
        type = float,
        required = True,
        help = "Threshold below which pixels of a given confidence are not counted in the final clump."
    )
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')