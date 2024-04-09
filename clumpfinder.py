from collections import deque

import torch
import torchvision

import typing as tp

import argparse

import src.data

def main(args:argparse.Namespace) -> bool:
    target_tensor = src.data.load_image(args.input_path, "L")
    clumps_list = find_clumps(target_tensor)
    print(clumps_list)
    return True


def iterative_flood_fill(matrix: torch.Tensor, x: int, y: int, visited: torch.Tensor) -> tp.List[tp.Tuple[int, int]]:
    queue = deque()
    queue.append((x, y))
    points = tp.List[tp.Tuple[int, int]]
    use_matrix = matrix
    print(use_matrix.shape)
    directions: tp.List[tp.Tuple[int, int]] = [(0,1), (1,0), (0,-1), (-1,0)]

    while queue:
        x, y = queue.pop()
        if x < 0 or x >= use_matrix.shape[0] or y >= use_matrix.shape[1]:
            continue
        if visited[x, y] or use_matrix[x, y] == 0:
            continue

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < use_matrix.shape[0] and 0 <= ny < use_matrix.shape[1]:
                queue.appendleft((nx, ny))
    return points

def find_clumps(matrix: torch.Tensor) -> tp.Dict[int, tp.List[tp.Tuple[int, int]]]:
    use_matrix = matrix[0]
    height, width = use_matrix.shape
    visited = torch.zeros_like(use_matrix, dtype=torch.bool)
    clumps: tp.Dict[int, tp.List[tp.Tuple[int, int]]] = {}
    clump_id = 0

    for x in range(height):
        for y in range(width):
            if use_matrix[x, y] > 0 and not visited[x, y]:
                clump_points = iterative_flood_fill(use_matrix, x, y, visited)
                if clump_points:
                    clumps[clump_id] = clump_points
                    clump_id += 1

def get_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path',
        type = str,
        required = True,
        help = 'Image to find chunks for.'
    )
    return parser


if __name__ == '__main__':
    args = get_argparser().parse_args()
    ok   = main(args)
    if ok:
        print('Done')