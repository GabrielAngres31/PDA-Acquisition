import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay, delaunay_plot_2d
import matplotlib.pyplot as plt
import pandas as pd
import os

filename = "C:/Users/Gabriel/Documents/GitHub/PDA-Acquisition/cleaned_images_default/tables/CLEAN_CON_MAX_R1_001uM_5dpg.lif - 1uMAZD_1_1_Merged.tif_AZD_INF_2025.output.png_.csv"

# points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                #    [2, 0], [2, 1], [2, 2]])
df = pd.read_csv(filename)

points = [list(t) for t in list(zip(df["centroid-0"], df["centroid-1"]))]
print(points)

# vor = Voronoi(points)
dela = Delaunay(points)

# print(vor.ridge_vertices)
# print(dela.simplices)
num_points = len(points) #.shape[0]
adj_matrix = np.zeros((num_points, num_points))



def getdist(coords, ind1, ind2):
    p1, p2 = coords[ind1], coords[ind2]
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

for simplex in dela.simplices:
    # print("grorg")
    for s in range(3): #simplex:
        i, j = simplex[s], simplex[(s+1)%3]
        # print([i,j])
        if not adj_matrix[i][j]:
            adj_matrix[i][j] = getdist(points, i, j)
            adj_matrix[j][i] = getdist(points, i, j)

print(np.rint(adj_matrix))

convex_hull_indices = list(set([x for y in list(dela.convex_hull) for x in y]))
print(convex_hull_indices)

closeness_linear = []
closeness_square = []
num_neighbors = []
for i in range(num_points):
    bits = adj_matrix[i]
    pieces = bits[bits != 0]
    closeness_linear.append(np.mean(pieces))
    closeness_square.append(np.sqrt(np.mean(np.square(pieces))))
    num_neighbors.append(len(pieces))

print([num_neighbors[i] for i in convex_hull_indices])


print(dela.points)
print(points == dela.points)
out_dela_df_data = [[points[i][0], 
                     points[i][1], 
                     np.where(adj_matrix[i] != 0)[0], 
                     len(np.where(adj_matrix[i] != 0)[0]), 
                     adj_matrix[i][adj_matrix[i] != 0],
                     (True if i in convex_hull_indices else False)] for i in range(len(points))]

out_dela_df = pd.DataFrame(data=out_dela_df_data, columns=["centroid-0", 
                                               "centroid-1", 
                                               "adj-indices", 
                                               "num_neighbors",
                                               "adj-distances_lin", 
                                               "is_hull"])

print(out_dela_df)
out_dela_df.to_csv(f"cleaned_images_default/qhulls/{os.path.basename(filename)}_QHULL.csv")


# print(closeness_linear)
# print(closeness_square)
# print(num_neighbors)



# adj_list = {i:[""] for i in range(len(points))}
# # print(adj_list)

# for tri in dela.simplices:
#     for i in range(3):
#         adj_list[tri[i]] = adj_list[tri[i]] + [tri[(i+1)%3]] + [tri[(i+2)%3]] 

# for i in range(len(points)):
#     adj_list[i] = list(set(adj_list[i][1:]))



# print(adj_list)

# fig = voronoi_plot_2d(vor)



# fig = delaunay_plot_2d(dela)

# plt.show()

