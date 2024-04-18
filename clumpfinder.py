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
    #print(target_tensor)
    clumps_list = find_clumps(target_tensor)
    
    if args.histogram:
        clump_info_dict:tp.Dict[int, int] = {}
        for id in clumps_list.keys():
            print(id)
            confidence_values = []
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

    if args.colorize:
        output_colorized = src.data.load_image(args.input_path, "RGB").permute(1, 2, 0)
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


def iterative_flood_fill(matrix: torch.Tensor, x: int, y: int, visited: torch.Tensor, threshold:float = 1.) -> tp.List[tp.Tuple[int, int]]:
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

def recursive_flood_fill(matrix: torch.Tensor, x: int, y: int, visited: torch.Tensor):
    use_matrix = matrix
    if x < 0 or x >- use_matrix.shape[0] or y >= use_matrix.shape[1]:
        pass    

def find_clumps(matrix: torch.Tensor) -> tp.Dict[int, tp.List[tp.Tuple[int, int]]]:
    use_matrix = matrix[0]
    height, width = use_matrix.shape
    visited = torch.zeros_like(use_matrix, dtype=torch.bool)
    clumps: tp.Dict[int, tp.List[tp.Tuple[int, int]]] = {}
    clump_id = 0

    for x in tqdm.tqdm(range(0, height)):
        for y in range(0, width):
            if use_matrix[x, y] > 0 and not visited[x, y]:
                clump_points = iterative_flood_fill(use_matrix, x, y, visited)
                if clump_points:
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
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')