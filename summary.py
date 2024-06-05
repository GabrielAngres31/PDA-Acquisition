# summary.py

import pandas as pd
import typing as tp
import matplotlib.pyplot as plt
import numpy as np


import seaborn as sns
import winsound
from pathlib import Path

from scipy.spatial import cKDTree
import os

def load_premade_table(path: str):
    return pd.read_csv(path)

def table_histogram(field: str, table: pd.DataFrame = None) -> None:
    return None

def table_scatterplot(fields: tp.List[tp.Tuple[str, str]]) -> None:
    return None

def table_3d_scatter(fields: tp.List[tp.Tuple[str, str, str]]) -> None:
    return None


# def cKDTree_method(data):
#     tree = cKDTree(data)
#     dists = tree.query(data, 2)
#     nn_dist = dists[0][:, 1]
#     return nn_dist

def dist_hist(path:str, num_neighbors: int) -> None:
    table: pd.DataFrame = load_premade_table(path)
    print("Getting Points...")
    # winsound.Beep(523, 800)
    points = [(x,y) for x in table['centroid-0'] for y in table['centroid-1']]
    print("Finding Distances...")
    # winsound.Beep(659, 800)
    tree = cKDTree(points)
    #dists = tree.query(points, 4)
    #nn_dist = dists[0][:, 1]
    nn_array = tree.query(points, num_neighbors+2)
    nn_dists = {}

    for i in range(num_neighbors+1):
        nn_dists[i+2] = nn_array[0][:, i+1]
    # lines = list(set([(a, b, q, r) for (a,b) in points for (q,r) in points]))
    #print("Crunching Dists...")
    # winsound.Beep(784, 800)
    #squares = [(a-q)^2+(b-r)^2 for (a, b, q, r) in lines]
    #dists = [np.sqrt(n) for n in squares]
    #print(nn_dists)
    for i in range(num_neighbors+1):
        file = open(f'{path[:-4]}_{i}.txt','w')
        for item in nn_dists[i+2]:
            file.write(f"{item}\n")
        file.close()
        print("Generating Histogram...")
        # winsound.Beep(np.random.random_integers(523, 2093), 800)
        #plt.hist(nn_dist)#, bins=list(range(0, 2000, 120)))
        #plt.xlim(0, 2000)
        plt.ylim(0,   80)
        sns.violinplot(nn_dists[i+2])
        plt.title(f"Histogram of NN ^{i}")
        #plt.figure(figsize = (8,6))
        os.makedirs(f"{Path(path).parent}\\NN_dists\\{Path(path).stem}", exist_ok = True)
        plt.savefig(f"{Path(path).parent}\\NN_dists\\{Path(path).stem}\\{Path(path).stem}_{i}.png")
        # plt.show()
        plt.clf()

    # winsound.Beep( 523, 150)
    # winsound.Beep( 659, 150)
    # winsound.Beep( 784, 150)
    # winsound.Beep(1046, 150)
    
    return None


# dist_hist("inference/tables/cotE02_STOMATA_MASKS.csv", 30)




