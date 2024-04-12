from collections import deque

import torch
import torchvision

import typing as tp

import argparse

import src.data
import tqdm

import PIL


def main(args:argparse.Namespace) -> bool:
    print(src.data.load_image(args.input_path, "RGB"))
    print(src.data.load_image(args.input_path, "L"))
    target_tensor = src.data.load_image(args.input_path, "L")
    clumps_list = find_clumps(target_tensor)
    #print(clumps_list)
    #print(clumps_list.keys())

    #TODO: You're not importing the image as RGB. FIND OUT WHY!!!!!

    print(clumps_list[0])
    if args.histogram:
        pass

    if args.colorize:
        #output_colorized = src.data.load_image(args.input_path, "RGBA")
        image = PIL.Image.open(args.input_path).convert("RGB")
        print(str(image))
        output_colorized = torchvision.transforms.PILToTensor(image)
        #output_colorized = src.data.load_inputimage(args.input_path)
        #clumps_iter = dict(zip(clumps_list.keys(), clumps_list.values()))
        print(clumps_list.keys())
        for id in clumps_list.keys():
            color = [(64+id*31)%256, (72+id*29)%256, (80+id*23)%256]
            for pixel in clumps_list[id]:
                print(output_colorized[0])
                print(output_colorized[0, pixel[0]])
                print(output_colorized[0, pixel[0], pixel[1]])
                
                output_colorized[0, pixel[0], pixel[1]] = torch.Tensor(*color, size=1)
                print(output_colorized[0, pixel[0], pixel[1]])

        filename_component = args.input_path.split("/")[-1]
        src.data.save_image(filename_component+".output.png", output_colorized)
    return True


def iterative_flood_fill(matrix: torch.Tensor, x: int, y: int, visited: torch.Tensor) -> tp.List[tp.Tuple[int, int]]:
    #print("Beep!")
    queue = deque()
    queue.append((x, y))
    points = list()
    use_matrix = matrix
    #print(use_matrix.shape)
    directions: tp.List[tp.Tuple[int, int]] = [(0,1), (1,0), (0,-1), (-1,0)]
    
    while queue:
        x, y = queue.pop()
        if x < 0 or x >= use_matrix.shape[0] or y >= use_matrix.shape[1]:
            continue
        if visited[x, y] or use_matrix[x, y] == 0:
            #print("argh!!!!!!")
            continue
        #print("Aight.")

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            #print(f"{y} {x}")
            if 0 <= nx < use_matrix.shape[0] and 0 <= ny < use_matrix.shape[1]:
                if (nx, ny) not in points:
                    queue.appendleft((nx, ny))
        visited[x, y] = True
        points.append(tuple([x,y]))
        #points.append(tuple([nx, ny]))
                    #print(len(queue))
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

    #TODO: Add pre-disqualifier for empty spaces
    def empty_space_remove(mat_check: torch.Tensor, visited_blank: torch.Tensor, step: int = 12):
        
        for x in range(0, height, step):
            for y in range(0, width, step):
                #print(mat_check[0,x,y])
                pass
    empty_space_remove(matrix, matrix)

    for x in tqdm.tqdm(range(0, height)):
        for y in range(0, width):
            if use_matrix[x, y] > 0 and not visited[x, y]:
                #print("Determining Clump...")
                #print(x,y)
                clump_points = iterative_flood_fill(use_matrix, x, y, visited)
                if clump_points:
                    #print(f"Located Clump #{clump_id}")
                    clumps[clump_id] = clump_points
                    #print(clumps[clump_id])
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