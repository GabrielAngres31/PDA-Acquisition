# voronoi_scrapbox

import numpy as np

import pandas as pd

from scipy.spatial import Voronoi, voronoi_plot_2d
import shapely.geometry
import shapely.ops

import matplotlib.pyplot as plt

# clumps_csv = pd.read_csv("./inference/clump_data/cot1_STOMATA_MASKS.csv")
# assert "centroid-0" in clumps_csv.columns and "centroid-1" in clumps_csv.columns, f"You don't have the centroid data in this dataframe! You have: {clumps_csv.columns}"
# vertices = clumps_csv[["centroid-1", "centroid-0"]]



# vor = Voronoi(vertices)
# voronoi_plot_2d(vor)
# img = plt.imread("SCD_training_data/source_images/BASE/cot1.tif")
# plt.imshow(img)
# plt.show()




import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# import numpy as np



clumps_csv = pd.read_csv("./inference/clump_data/cot1_STOMATA_MASKS.csv")
assert "centroid-0" in clumps_csv.columns and "centroid-1" in clumps_csv.columns, f"You don't have the centroid data in this dataframe! You have: {clumps_csv.columns}"
vertices = clumps_csv[["centroid-1", "centroid-0"]]

# print(vertices.values.tolist())

img = plt.imread("SCD_training_data/source_images/BASE/cot1.tif")
plt.imshow(img)


for [x,y] in vertices.values.tolist():
    inner_radius = 0
    outer_radius = 160
    center_x = x
    center_y = y
    halo_color = 'gray'

    # center_color = 'none' # for an empty center
    center_color = '#ff334466'  ## redish with 25% alpha
    cmap = LinearSegmentedColormap.from_list('', ['#FFFFFF00', halo_color])
    cmap.set_bad(center_color)

    xmin = center_x - outer_radius
    xmax = center_x + outer_radius
    ymin = center_y - outer_radius
    ymax = center_y + outer_radius
    x, y = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    z = np.where(r < inner_radius, np.nan, np.clip(outer_radius - r, 0, np.inf))
    plt.imshow(z, cmap=cmap, extent=[xmin, xmax, ymin, ymax], origin='lower', zorder=3)

plt.axis('equal')
plt.show()